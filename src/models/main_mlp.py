from flax import nnx
import jax
import jax.numpy as jnp
import tensorflow as tf 
from typing import Callable
import optax 
import matplotlib.pyplot as plt

# TODO add param for dropout rate
class MLP(nnx.Module):
    def __init__(self, 
                 hidden_dims, 
                 rngs: nnx.Rngs, 
                 activation: Callable = nnx.relu):
      
      self.mlp = nnx.Sequential(
         nnx.Linear(hidden_dims, hidden_dims, rngs=rngs),
         activation,
         nnx.Dropout(rate=.1, rngs=rngs),
         nnx.Linear(hidden_dims, hidden_dims, rngs=rngs),
         nnx.Dropout(rate=.1, rngs=rngs)
      )

    def __call__(self, x):
        x = self.mlp(x)
        return x

class MLPModel(nnx.Module):
    def __init__(self, 
                 in_dims, 
                 hidden_dims, 
                 out_dims, 
                 num_mlps, 
                 rngs: nnx.Rngs, 
                 activation: Callable = nnx.relu):
        
        self.hidden_dims = hidden_dims
        self.initial_linear = nnx.Linear(in_dims, hidden_dims, rngs=rngs)
        self.activation = activation 
        self.out_layer = nnx.Linear(hidden_dims, out_dims, rngs=rngs)
        keys = jax.random.split(jax.random.key(0), num_mlps)
        self.models = self.create_models(keys)
    
    # TODO more flexible way of writing this? 
    @nnx.vmap(in_axes=(None, 0), out_axes=0)
    def create_models(self, key: jax.Array):
        return MLP(hidden_dims=self.hidden_dims, rngs=nnx.Rngs(key))

    def __call__(self, x):
        @nnx.scan(in_axes=(0, nnx.Carry), out_axes=nnx.Carry)
        def forward(model: MLP, x):
            x = model(x)
            return x 
        z = self.activation(self.initial_linear(x))
        z = forward(self.models, z)
        z_out = self.out_layer(z)
        return z_out
    
    def fit(self, batch_size, train_steps, dataset: tf.data.Dataset):
        """
        We train the MLP model such that it takes 
        """
        dataset = dataset.repeat().shuffle(150)
        dataset = dataset.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)

if __name__ == '__main__':
    rngs = nnx.Rngs(0)
    in_dims = 41
    hidden_dims = 10
    out_dims = 20
    num_mlps = 1
    learning_rate = 0.005
    momentum = 0.9
    train_ratio = .75
    batch_size = 32 
    train_steps = 5000
    eval_every = 5
    shuffle_buffer_size = 1024

    model = MLPModel(in_dims, hidden_dims, out_dims, num_mlps, rngs)
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))

    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average('loss'),
    )

    def loss_fn(model: MLPModel, batch):
        logits = model(batch['inputs'])
        loss = optax.softmax_cross_entropy(
            logits=logits, labels=batch['targets']
        ).mean()
        return loss, logits

    @nnx.jit
    def train_step(model: MLPModel,
                   optimizer: nnx.Optimizer,
                   metrics: nnx.MultiMetric,
                   batch):
        
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(model, batch)

        # convert one-hot to class indices for the metric
        # TODO: dtype here?
        labels_idx = jnp.argmax(batch['targets'], axis=-1)
        # jax.debug.print("labels look like {labels_idx}", labels_idx=labels_idx)
        metrics.update(loss=loss,
                       logits=logits,
                       labels=labels_idx)      
        
        optimizer.update(grads)

    @nnx.jit
    def eval_step(model: MLPModel, metrics: nnx.MultiMetric, batch):
        loss, logits = loss_fn(model, batch)
        labels_idx = jnp.argmax(batch['targets'], axis=-1)
        metrics.update(loss=loss, logits=logits, labels=labels_idx)  # In-place updates.

    metrics_history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': [],
    }
    
    mlp_ds = tf.data.Dataset.load('./data/mlp_tf_dataset')
    num_data_points = tf.data.experimental.cardinality(mlp_ds).numpy()
    mlp_ds = mlp_ds.shuffle(num_data_points)

    num_train_points = int(num_data_points * train_ratio)
    train_ds = mlp_ds.take(num_train_points)
    test_ds = mlp_ds.skip(num_train_points)

    train_ds = train_ds.repeat().shuffle(shuffle_buffer_size)
    train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

    plt.ion()  # interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 1) Add a big title for the whole figure:
    fig.suptitle("MLP Training Metrics", fontsize=16)

    # 2) Set titles and axis labels on each subplot:
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")

    # two empty Line2D objects we’ll update on every eval
    train_loss_line, = ax1.plot([], [], label="train_loss")
    test_loss_line,  = ax1.plot([], [], label="test_loss")
    train_acc_line,  = ax2.plot([], [], label="train_accuracy")
    test_acc_line,   = ax2.plot([], [], label="test_accuracy")

    ax1.legend()
    ax2.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave room for suptitle
    fig.show()

    for step, batch in enumerate(train_ds.as_numpy_iterator()):
        train_step(model, optimizer, metrics, batch)

        if step > 0 and (step % eval_every == 0 or step == train_steps - 1):
            # … compute & update metrics_history as before …

            # ----- update stored y‑data -----
            epochs = list(range(1, len(metrics_history['train_loss']) + 1))
            train_loss_line.set_data(epochs, metrics_history['train_loss'])
            test_loss_line.set_data( epochs, metrics_history['test_loss'])
            train_acc_line.set_data(  epochs, metrics_history['train_accuracy'])
            test_acc_line.set_data(   epochs, metrics_history['test_accuracy'])

            # rescale the axes so new points are visible
            for ax in (ax1, ax2):
                ax.relim()
                ax.autoscale_view()

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

            # 3) Save out the figure for this epoch:
            current_epoch = len(metrics_history['train_loss'])
            fig.savefig(f"mlp_metrics_epoch_{current_epoch:03d}.png", dpi=150)



