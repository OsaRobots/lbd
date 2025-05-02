from flax import nnx
import jax 
from jax import random, numpy as jnp
import tensorflow as tf 
from typing import Callable
import optax 
import matplotlib.pyplot as plt

# TODO: n_layers?
# TODO: Training accuracy seems suspiciously good...
# Ensure the mask is defined properly

class MLP(nnx.Module):
    def __init__(self, hidden_dims, num_heads, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(hidden_dims, hidden_dims * num_heads, rngs=rngs) # Typical expansion
        self.dense2 = nnx.Linear(hidden_dims * num_heads, hidden_dims, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs) # Add dropout consistent with Transformer block

    def __call__(self, x, *, deterministic: bool = False):
        x = self.dense1(x)
        x = jax.nn.relu(x) # Or gelu
        x = self.dense2(x)
        x = self.dropout(x, deterministic=deterministic)
        return x
    
class TransformerModel(nnx.Module):
    def __init__(self,
                 in_dims,
                 hidden_dims,
                 out_dims,
                 num_heads,
                 rngs: nnx.Rngs):

        # TODO: Is this even a normal-ish Transformer implemenation?
        # - It's a single transformer block (pre-LN variant).
        # - Lacks positional embeddings, which are usually crucial for sequence order.
        # - Typically multiple blocks are stacked.

        self.rngs = rngs 
        self.initial_proj = nnx.Linear(in_dims, hidden_dims, rngs=rngs)
        self.ln1 = nnx.LayerNorm(num_features=hidden_dims, rngs=rngs)
        # keep dim. the same for residual stream
        self.mha = nnx.MultiHeadAttention(num_heads=num_heads,
                                          in_features=hidden_dims,
                                          qkv_features=hidden_dims, # Often qkv_features = hidden_dims // num_heads
                                          out_features=hidden_dims,
                                          dropout_rate=.1, 
                                          rngs=rngs)

        self.dropout_res1 = nnx.Dropout(rate=.1, rngs=rngs) # Dropout for the first residual connection
        self.ln2 = nnx.LayerNorm(num_features=hidden_dims, rngs=rngs) # LayerNorm on the *output* of the residual stream
        self.mlp = MLP(hidden_dims, num_heads=num_heads, rngs=rngs)
        self.dropout_res2 = nnx.Dropout(rate=.1, rngs=rngs) # Dropout for the second residual connection
        self.action_logits = nnx.Linear(hidden_dims, out_dims, rngs=rngs)
        
          
    def __call__(self,
                inputs: jax.Array,
                # rngs: nnx.Rngs, 
                *,
                deterministic: bool = False):

        # NOTE: Teacher-forcing is handled by the data prep 
        if inputs.ndim != 3:
             raise ValueError(f"Expected inputs to have 3 dimensions (batch, seq, features), got {inputs.shape}")
        causal_mask = nnx.make_causal_mask(inputs[:, :, 0]) # use any feature slice just to get batch/seq dims

        # TODO: positional embeddings, rngs 
        x_proj = self.initial_proj(inputs)
        x_norm = self.ln1(x_proj)
        attn_output = self.mha(x_norm, mask=causal_mask, decode=False, deterministic=deterministic)
        x = x_proj + self.dropout_res1(attn_output, deterministic=deterministic)

        z_norm = self.ln2(x) 
        mlp_output = self.mlp(z_norm, deterministic=deterministic)
        z = x + self.dropout_res2(mlp_output, deterministic=deterministic)
        logits = self.action_logits(z)
        return logits

if __name__ == '__main__':
    rngs = nnx.Rngs(params=0, dropout=random.key(1))
    in_dims = 62
    hidden_dims = 128 # Often larger than num_heads * head_dim
    out_dims = 20 # Number of arms/actions
    num_heads = 4
    learning_rate = 0.001 # AdamW often uses smaller LR
    # momentum = 0.9 # AdamW uses betas, not momentum directly
    weight_decay = 0.01 # Common for AdamW

    train_ratio = .75
    batch_size = 32 # Adjust based on GPU memory
    train_steps = 5000 # Increased steps might be needed
    eval_every = 100 # Evaluate less frequently if steps increase
    shuffle_buffer_size = 256 # Larger buffer for better shuffling
    sequence_length = 100 # Assuming MAB task has T=100 trials per subject based on data prep

    model = TransformerModel(in_dims, hidden_dims, out_dims, num_heads, rngs=rngs)
    # Use AdamW as it's common for Transformers
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, weight_decay=weight_decay))

    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average('loss'),
    )

    def loss_fn(model: TransformerModel, batch):
        # TODO: do I need to mask anything?
        logits = model(batch['inputs'])
        loss = optax.softmax_cross_entropy(
            logits=logits, labels=batch['targets']
        ).mean()
        return loss, logits

    @nnx.jit
    def train_step(model: TransformerModel,
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
    def eval_step(model: TransformerModel, metrics: nnx.MultiMetric, batch):
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


    