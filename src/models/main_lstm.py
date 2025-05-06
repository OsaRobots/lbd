from flax import nnx
import jax
import jax.numpy as jnp
import tensorflow as tf 
from typing import Callable
import optax 
import matplotlib.pyplot as plt
import pandas as pd 

# TODO: n_layers?

class LSTMModel(nnx.Module):
    def __init__(self, in_dims, hidden_dims, out_dims, rngs: nnx.Rngs):
        #self.lstm_cell = nnx.OptimizedLSTMCell(in_dims, hidden_dims, rngs=rngs)
        self.lstm = nnx.RNN(cell=nnx.OptimizedLSTMCell(in_dims, hidden_dims, rngs=rngs), rngs=rngs)
        self.head = nnx.Linear(hidden_dims, out_dims, rngs=rngs)
        # self.mlp = nnx.Sequential(nnx.Linear(hidden_dims, hidden_dims, rngs=rngs), 
        #                           nnx.relu, 
        #                           nnx.Linear(hidden_dims, out_dims, rngs=rngs))      
          
    def __call__(self, xs):
        # NOTE: no need to worry about teacher-forcing because we already give the true action in the inputs. 
        zs = self.lstm(xs) # shape (b_size, time, hidden_dims)
        logits = self.head(zs) # shape: (b_size, time, out_dims)
        return logits 

if __name__ == '__main__':
    rngs = nnx.Rngs(0)
    in_dims = 62
    hidden_dims = 128
    out_dims = 20
    learning_rate = 0.001
    weight_decay = 0.01

    momentum = 0.9
    train_ratio = .75
    batch_size = 32 
    train_steps = 2500
    eval_every = 100
    shuffle_buffer_size = 256

    model = LSTMModel(in_dims, hidden_dims, out_dims, rngs)
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum, weight_decay=weight_decay))

    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average('loss'),
    )

    def loss_fn(model: LSTMModel, batch):
        # TODO: do I need to mask anything?
        logits = model(batch['inputs'])
        loss = optax.softmax_cross_entropy(
            logits=logits, labels=batch['targets']
        ).mean()
        return loss, logits

    @nnx.jit
    def train_step(model: LSTMModel,
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
    def eval_step(model: LSTMModel, metrics: nnx.MultiMetric, batch):
        loss, logits = loss_fn(model, batch)
        labels_idx = jnp.argmax(batch['targets'], axis=-1) # is -1 sill fine? should be...
        metrics.update(loss=loss, logits=logits, labels=labels_idx)  # In-place updates.

    metrics_history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': [],
    }
    
    rnn_ds = tf.data.Dataset.load('./data/rnn_tf_dataset')
    num_data_points = tf.data.experimental.cardinality(rnn_ds).numpy()
    rnn_ds = rnn_ds.shuffle(num_data_points)

    num_train_points = int(num_data_points * train_ratio)
    train_ds = rnn_ds.take(num_train_points)
    test_ds = rnn_ds.skip(num_train_points)

    train_ds = train_ds.repeat().shuffle(shuffle_buffer_size)
    train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

    plt.ion() # interactive mode for now 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.set_title("Loss")
    ax2.set_title("Accuracy")

    fig.suptitle("LSTM Training Metrics", fontsize=16)

    # 2) Set titles and axis labels on each subplot:
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")

    # two empty Line2D objects we’ll update on every eval
    train_loss_line,  = ax1.plot([], [], label="train_loss")
    test_loss_line,   = ax1.plot([], [], label="test_loss")
    train_acc_line,   = ax2.plot([], [], label="train_accuracy")
    test_acc_line,    = ax2.plot([], [], label="test_accuracy")

    ax1.legend()
    ax2.legend()
    
    fig.tight_layout()
    fig.show()

    for step, batch in enumerate(train_ds.as_numpy_iterator()):
        train_step(model, optimizer, metrics, batch)

        if step > 0 and (step % eval_every == 0 or step == train_steps - 1):  # One training epoch has passed.
        # Log the training metrics.
            for metric, value in metrics.compute().items():  # Compute the metrics.
                metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
            metrics.reset()  # Reset the metrics for the test set.

        # Compute the metrics on the test set after each training epoch.
            for test_batch in test_ds.as_numpy_iterator():
                eval_step(model, metrics, test_batch)

        # Log the test metrics.
            for metric, value in metrics.compute().items():
                metrics_history[f'test_{metric}'].append(value)
            metrics.reset()  # Reset the metrics for the next training epoch.

        # Plot loss and accuracy in subplots
            # ----- update stored y‑data -----
            train_loss_line.set_data(range(len(metrics_history['train_loss'])),
                                    metrics_history['train_loss'])
            test_loss_line.set_data(range(len(metrics_history['test_loss'])),
                                    metrics_history['test_loss'])
            train_acc_line.set_data(range(len(metrics_history['train_accuracy'])),
                                    metrics_history['train_accuracy'])
            test_acc_line.set_data(range(len(metrics_history['test_accuracy'])),
                                metrics_history['test_accuracy'])

            # rescale the axes so new points are visible
            for ax in (ax1, ax2):
                ax.relim()              # recompute limits
                ax.autoscale_view()     # apply them

            fig.canvas.draw_idle()      # queue a repaint
            fig.canvas.flush_events()   # force the GUI event loop to process it
            plt.pause(0.001)            # tiny sleep keeps things responsive

        epochs = list(range(1, len(metrics_history['train_accuracy']) + 1))
        df = pd.DataFrame({
            'epoch': epochs,
            'train_accuracy': metrics_history['train_accuracy'],
            'test_accuracy':  metrics_history['test_accuracy'],
        })

        # write it out
        df.to_csv("lstm_accuracy.csv", index=False)

    