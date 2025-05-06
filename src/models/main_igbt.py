# Inverse Generalized Belief Transport 
from functools import partial 
from flax import nnx
import jax 
from jax import lax
from typing import Dict, Tuple 
from jax import random, numpy as jnp
import tensorflow as tf 
import optax 
import matplotlib.pyplot as plt
from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
import pandas as pd 

class CostNet(nnx.Module):
    def __init__(self, in_dims, hidden_dims, out_dims, rngs: nnx.Rngs):
        self.mlp = nnx.Sequential(
            nnx.Linear(in_dims,  hidden_dims, rngs=rngs), nnx.relu,
            nnx.Linear(hidden_dims, hidden_dims, rngs=rngs), nnx.relu,
            nnx.Linear(hidden_dims, out_dims, rngs=rngs),               # K costs
        )
        # self.eps_gamma = nnx.Param(value=1.)
        # self.eps_theta = nnx.Param(value=1.)
        # self.beta = nnx.Param(value=1.)
        self.eps_gamma = 0.05       # static, safe
        self.eps_theta = 0.10
        self.beta      = 10.0

    def __call__(self, xs):                  
        return self.mlp(xs)                 

    def loss(self, batch):
        """
        batch['inputs']  : (B, T, in_dim)
        batch['targets'] : (B, T, K)   one‑hot
        batch['prior']   : (B, K)      θ₀  (posterior from previous sequence)
        """
        costs_seq = self(batch["inputs"])          # (B, T, K)

        # put time axis first so lax.scan iterates over it
        costs_TBK   = jnp.swapaxes(costs_seq, 0, 1)      # (T, B, K)
        targets_TBK = jnp.swapaxes(batch["targets"], 0, 1)

        def step(theta_prev, inp):
            costs_t, targ_t = inp                      # each (B, K)
            post_t = self.sinkhorn_posterior(costs_t, theta_prev)
            log_pi = jax.nn.log_softmax(self.beta * post_t, axis=-1)
            nll_t  = -(log_pi * targ_t).sum(axis=-1)   # (B,)
            return post_t, (nll_t, post_t)

        _, (nlls, posts) = lax.scan(
            step,
            batch["prior"],                           # θ₀  (B, K)
            (costs_TBK, targets_TBK)                  # sequence inputs
        )

        loss  = nlls.mean()           # averaged over batch & time
        final = posts[-1]             # θ_T  (B, K)
        return loss, final   
    
    def sinkhorn_posterior(self, costs, prior, iters=50):
        # ── 1.  constants, not tracers  ──────────────────────────
        # eps_p     = float(self.eps_gamma.value)  
        # eps_theta = float(self.eps_theta.value)   
        eps_p = self.eps_gamma
        eps_theta = self.eps_theta
        tau_b = 1.0 - eps_theta / (eps_theta + eps_p)

        # ── 2.  vmap one Sinkhorn per row  ───────────────────────
        def single_row(cost_row, prior_row):
            g = geometry.Geometry(cost_matrix=cost_row[None, :], epsilon=eps_p)
            prob = linear_problem.LinearProblem(
                g,
                a=jnp.array([1.0]),          # row mass
                b=prior_row,                 # (K,)
                tau_a=1.0,
                tau_b=tau_b,                 # python float → static
            )
            gamma = sinkhorn.Sinkhorn(max_iterations=iters, threshold=1e-4)(prob).matrix
            return gamma[0] / gamma.sum()            # (K,)

        return jax.vmap(single_row)(costs, prior)   # (B,K)

@nnx.jit
def train_step(model: CostNet,
               optimizer: nnx.Optimizer,
               metrics: nnx.MultiMetric,
               batch):

    # --- compute grads wrt the model ------------------------
    def loss_fn(m):                     # m is CostNet
        return m.loss(batch)            # returns (loss, aux)

    (loss, posterior_T), grads = nnx.value_and_grad(
        loss_fn, has_aux=True)(model)

    # --- metrics -------------------------------------------
    last_labels = jnp.argmax(batch["targets"][:, -1, :], axis=-1)
    metrics.update(loss=loss, logits=posterior_T, labels=last_labels)

    # --- optax update (grads now numeric) ------------------
    optimizer.update(grads)
    return posterior_T

@nnx.jit
def eval_step(model: CostNet, metrics: nnx.MultiMetric, batch):
    loss, logits = model.loss(batch)
    labels_idx = jnp.argmax(batch['targets'][:, -1, :], axis=-1)  # (B,)
    metrics.update(loss=loss, logits=logits, labels=labels_idx)  # In-place updates.

if __name__ == '__main__':
    rngs          = nnx.Rngs(0)
    in_dims       = 62
    hidden_dims   = 128
    K             = 20        
    learning_rate = 1e-3
    weight_decay  = 0.01

    batch_size = 32
    out_dims = 20
    theta_prev = jnp.full((batch_size, out_dims), 1/out_dims)  

    model = CostNet(in_dims, hidden_dims, K, rngs)
    optim = nnx.Optimizer(model, optax.adamw(learning_rate, weight_decay=weight_decay))

    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average('loss'),
    )

    train_ratio = .75
    batch_size = 32
    train_steps = 2500
    eval_every = 100
    shuffle_buf = 256

    rnn_ds = tf.data.Dataset.load('./data/rnn_tf_dataset')
    n_points = tf.data.experimental.cardinality(rnn_ds).numpy()
    rnn_ds = rnn_ds.shuffle(n_points)

    n_train = int(n_points * train_ratio)
    train_ds = rnn_ds.take(n_train)
    test_ds = rnn_ds.skip(n_train)

    train_ds = train_ds.repeat().shuffle(shuffle_buf).batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    ax1.set_title("Loss");   ax2.set_title("Accuracy")
    train_loss_ln ,= ax1.plot([], [], label="train_loss")
    test_loss_ln  ,= ax1.plot([], [], label="test_loss")
    train_acc_ln  ,= ax2.plot([], [], label="train_acc")
    test_acc_ln   ,= ax2.plot([], [], label="test_acc")
    ax1.legend(); ax2.legend(); fig.tight_layout(); fig.show()

    fig.suptitle("IGBT Training Metrics", fontsize=16)

    # 2) Set titles and axis labels on each subplot:
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")

    metrics_hist = {k: [] for k in
                    ['train_loss','train_accuracy','test_loss','test_accuracy']}

    theta_prev = jnp.full((batch_size, K), 1/K)            

    for step, raw in enumerate(train_ds.as_numpy_iterator()):
        batch = { 'inputs' : jnp.asarray(raw['inputs']),    # shape (B,62)
                'targets': jnp.asarray(raw['targets']),   # one‑hot (B,K)
                'prior'  : theta_prev }                   # (B,K)

        posterior = train_step(model, optim, metrics, batch)

        # ---- carry posterior forward as next prior (NO grad) ----
        theta_prev = posterior

        # ---- periodic test & plot (unchanged) -------------------
        if step>0 and (step%eval_every==0 or step==train_steps-1):
            for m,v in metrics.compute().items():
                metrics_hist[f'train_{m}'].append(v)
            metrics.reset()

            # test set
            for test_raw in test_ds.as_numpy_iterator():
                test_b = {
                    'inputs':  jnp.asarray(test_raw['inputs']),
                    'targets': jnp.asarray(test_raw['targets']),
                    'prior'  : theta_prev,              # any prior works for eval
                }
                eval_step(model, metrics, test_b)
            for m,v in metrics.compute().items():
                metrics_hist[f'test_{m}'].append(v)
            metrics.reset()

            # update plots
            train_loss_ln.set_data(range(len(metrics_hist['train_loss'])),
                                metrics_hist['train_loss'])
            test_loss_ln.set_data(range(len(metrics_hist['test_loss'])),
                                metrics_hist['test_loss'])
            train_acc_ln.set_data(range(len(metrics_hist['train_accuracy'])),
                                metrics_hist['train_accuracy'])
            test_acc_ln.set_data(range(len(metrics_hist['test_accuracy'])),
                                metrics_hist['test_accuracy'])
            for ax in (ax1,ax2):
                ax.relim(); ax.autoscale_view()
            fig.canvas.draw_idle(); fig.canvas.flush_events(); plt.pause(0.001)

    # print("Finished training.")

    epochs = list(range(1, len(metrics_hist['train_accuracy']) + 1))
    df = pd.DataFrame({
            'epoch': epochs,
            'train_accuracy': metrics_hist['train_accuracy'],
            'test_accuracy':  metrics_hist['test_accuracy'],
        })

        # write it out
    df.to_csv("igbt_accuracy.csv", index=False)
