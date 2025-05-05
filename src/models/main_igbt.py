# Inverse Generalized Belief Transport 

from flax import nnx
import jax 
from jax import random, numpy as jnp
import tensorflow as tf 
import optax 
import matplotlib.pyplot as plt
import ott.geometry as geom
import ott.problems.linear as lin
import ott.solvers.linear as linsol

K = 20                     
feat_dim = 62     
batch_size = 32
train_steps = 5000

class BanditCost(nnx.Module):
    """c_phi(d, j)  where  d = (x, a, r)."""
    def __init__(self, hidden_dims, rngs: nnx.Rngs):
        self.net = nnx.Sequential(
            nnx.Linear(hidden_dims, hidden_dims, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dims, hidden_dims, rngs=rngs),
            nnx.relu,
        )
        # one scalar per arm j
        self.head = nnx.Linear(hidden_dims, out_dims=K, rngs=rngs)

    def __call__(self, datum):
        # datum shape (B, feat_dim)  ->  costs shape (B, K)
        h = self.net(datum)
        return self.head(h)            # no softmax; raw costs
    
def ot_posterior(costs,               # (B, K)  output of BanditCost
                 prior,               # (B, K)  theta_{t-1}
                 eps_p=0.05,
                 eps_eta=0.0,         # row KL → 0 because η_t is a Dirac
                 eps_theta=0.1):
    """
    One Sinkhorn solve per datum in the batch.
    returns   posterior θ_t  shape (B, K)
    """
    # ott needs 2‑D arrays even for “one row” problems
    geoms = geom.geometry(
        cost_matrix=costs,            # C_ij
        epsilon=eps_p,
        dtype=costs.dtype,
    )

    # unbalanced KL weights on the two marginals
    problem = lin.linear_problem(
        geoms,
        a=None,                       # let ott treat rows as free ⇒ KL row=0
        b=prior,                      # soft column constraint
        tau_a=1.0 - eps_eta / (eps_eta + eps_p),
        tau_b=1.0 - eps_theta / (eps_theta + eps_p),
    )
    
    solver = linsol.sinkhorn(max_iterations=50, threshold=1e-4)
    out = solver(problem)            # .gm is the transport plan

    gamma = out.matrix               # shape (B, K)
    posterior = gamma / gamma.sum(axis=1, keepdims=True)
    return posterior                 # differentiable w.r.t. costs & prior
    
def choice_log_probs(theta, beta=10.0):
    """Softmax policy π_β(a | θ)."""
    return jax.nn.log_softmax(beta * theta, axis=-1)

def inverse_uot_loss(cost_net: BanditCost,
                     batch,              # expects keys 'datum', 'prior', 'action'
                     eps_p, eps_theta, beta=10.0):
    costs = cost_net(batch['datum'])          # (B, K)
    posterior = ot_posterior(costs,
                             batch['prior'],
                             eps_p=eps_p,
                             eps_theta=eps_theta)   # (B, K)
    log_pi = choice_log_probs(posterior, beta)      # (B, K)
    loglik = jnp.take_along_axis(
        log_pi, batch['action'][..., None], axis=-1
    ).squeeze(-1)                                 # (B,)
    return -loglik.mean()                         # negative LL

# ─── initialise modules ───────────────────────────────────────────────────────
if __name__ == '__main__':
    rngs = nnx.Rngs(0)
    cost_net = BanditCost(hidden_dims=128, rngs=rngs)
    opt = nnx.Optimizer(cost_net, optax.adamw(1e-3, weight_decay=1e-2))
    
    eps_p      = nnx.Variable(init=0.05)   # log‑parameterise if you want them learned
    eps_theta  = nnx.Variable(init=0.1)
    beta_param = nnx.Variable(init=2.0)     # log‑temperature for policy

    def to_features(elem):
        r   = tf.expand_dims(elem["reward"], -1)             # (1,)
        aOH = tf.one_hot(elem["arm"], depth=K)               # (K,)
        x   = elem["image_feat"]                             # (40,)
        datum = tf.concat([r, aOH, x], axis=-1)              # (feat_dim,)
        return {
            "datum" : datum,
            "action": elem["arm"],
        }
    
    ds = tf.data.Dataset.load("./data/rnn_tf_dataset")
    num_points = tf.data.experimental.cardinality(ds).numpy()
    batch_size = 32

    ds = ds.map(to_features).shuffle(1024).batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(2)                         # overlap CPU → GPU transfer

    # -------------------------------------------------------------------------
    # 2)  Wrap as a Python generator that also injects the running prior θ_{t-1}
    # -------------------------------------------------------------------------
    def make_data_iterator(tf_dataset, k_arms):
        """Yields dicts with keys datum, action, prior."""
        theta_prev = jnp.full((batch_size, k_arms), 1.0 / k_arms)   # uniform start
        for elem in tf_dataset.as_numpy_iterator():                 # numpy → host
            batch = {
                "datum" : jnp.asarray(elem["datum"]),               # (B, feat_dim)
                "action": jnp.asarray(elem["action"]),              # (B,)
                "prior" : theta_prev,                               # (B, K)
            }
            yield batch
            # ---- update θ_prev for *next* batch, without tracking gradients ----
            with jax.disable_jit():                                 # or nnx.no_grad()
                costs   = cost_net(batch["datum"])                  # forward once
                theta_prev = ot_posterior(costs, theta_prev,
                                        eps_p.value, eps_theta.value)

    data_iterator = make_data_iterator(ds.repeat(), K)              # infinite stream

    @nnx.jit
    def train_step(net, opt, batch):
        def loss_fn(model, batch):
            return inverse_uot_loss(model, batch,
                                    eps_p.value,
                                    eps_theta.value,
                                    beta=jnp.exp(beta_param.value))
        (loss), grads = nnx.value_and_grad(loss_fn)(net, batch)
        opt.update(grads)
        return loss
    
    θ_prev = jnp.full((batch_size, K), 1.0 / K)   # uniform start

    for step, raw in enumerate(data_iterator):
        # build batch dict
        batch = {
            'datum' : raw['datum'],               # (B, feat_dim)
            'action': raw['action'],              # (B,)  int arm index
            'prior' : θ_prev,                     # (B, K)
        }

        loss = train_step(cost_net, opt, batch)

        # ---- update prior for next step ----
        with nnx.no_grad():
            costs     = cost_net(batch['datum'])
            θ_post    = ot_posterior(costs, θ_prev,
                                    eps_p.value,
                                    eps_theta.value)
        θ_prev = θ_post                           # feed to next datum
