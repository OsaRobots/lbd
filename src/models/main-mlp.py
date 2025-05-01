from flax import nnx
import jax
import jax.numpy as jnp
import tensorflow as tf 
from typing import Callable
import optax 

class MLP(nnx.Module):
    def __init__(self, 
                 hidden_dims, 
                 rngs: nnx.Rngs, 
                 activation: Callable = nnx.relu):
      
      self.mlp = nnx.Sequential(
         nnx.Linear(hidden_dims, hidden_dims, rngs=rngs),
         activation,
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
        labels_idx = jnp.argmax(batch['targets'], axis=-1)

        metrics.update(loss=loss,
                       logits=logits,
                       labels=labels_idx)      
        
        optimizer.update(grads)

    @nnx.jit
    def eval_step(model: MLPModel, metrics: nnx.MultiMetric, batch):
        loss, logits = loss_fn(model, batch)
        metrics.update(loss=loss, logits=logits, labels=batch['targets'])  # In-place updates.

    metrics_history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': [],
    }
    
    mlp_dataset = tf.data.Dataset.load('./data/mlp_tf_dataset')
    mlp_dataset = mlp_dataset.repeat().shuffle(150)
    mlp_dataset = mlp_dataset.batch(2, drop_remainder=True).take(50).prefetch(1)

    for step, batch in enumerate(mlp_dataset.as_numpy_iterator()):
        train_step(model, optimizer, metrics, batch)
        break 


