# Inverse Generalized Belief Transport 
from functools import partial 
from flax import nnx
import jax 
from typing import Dict, Tuple 
from jax import random, numpy as jnp
import tensorflow as tf 
import optax 
import matplotlib.pyplot as plt
from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

class CostNet(nnx.Module):
    def __init__(self, in_dims, hidden_dims, out_dims, rngs: nnx.Rngs):
        self.mlp = nnx.Sequential(
            nnx.Linear(in_dims,  hidden_dims, rngs=rngs), nnx.relu,
            nnx.Linear(hidden_dims, hidden_dims, rngs=rngs), nnx.relu,
            nnx.Linear(hidden_dims, out_dims, rngs=rngs),               # K costs
        )
        self.eps_gamma = nnx.Param(value=1.)
        self.eps_theta = nnx.Param(value=1.)
        self.beta = nnx.Param(value=1.)

    def __call__(self, xs):              # xs: (B, in_dims)
        return self.mlp(xs)              # (B, K)        
    
    def sinkhorn_posterior(self, costs, prior, iters=50):
        eps_p = self.eps_gamma.value
        eps_theta = self.eps_theta.value

        g = geometry.Geometry(cost_matrix=costs, epsilon=eps_p)
        tau_b = 1.0 - eps_theta / (eps_theta + eps_p)
        row_mass = jnp.ones(costs.shape[0])               # (B,)
        prob = linear_problem.LinearProblem(g,
                                            a=row_mass,
                                            b=prior,
                                            tau_a=1.0,
                                            tau_b=tau_b)
        gamma = sinkhorn.Sinkhorn(max_iterations=iters, threshold=1e-4)(prob).matrix
        return gamma / gamma.sum(axis=1, keepdims=True)

    def loss(self, batch):
        costs = self(batch['inputs'])
        posterior = self.sinkhorn_posterior(costs, batch['prior'])
        log_pi = jax.nn.log_softmax(self.beta * posterior, axis=-1)
        loss = -(log_pi * batch["targets"]).sum(axis=-1).mean()
        return loss, posterior

@nnx.jit
def train_step(model: CostNet, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    
    grad_fn = nnx.value_and_grad(model.loss, has_aux=True)
    (loss, posterior), grads = grad_fn(batch)
    labels_idx = jnp.argmax(batch['targets'], axis=-1)
    # jax.debug.print("labels look like {labels_idx}", labels_idx=labels_idx)
    metrics.update(loss=loss,
                    logits=posterior,
                    labels=labels_idx)      
    
    optimizer.update(grads)
    return posterior 

@nnx.jit
def eval_step(model: CostNet, metrics: nnx.MultiMetric, batch):
    loss, logits = model.loss(batch)
    labels_idx = jnp.argmax(batch['targets'], axis=-1) # is -1 sill fine? should be...
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
    train_steps = 5000
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
                eval_step(optim, metrics, test_b)
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

    print("Finished training.")
