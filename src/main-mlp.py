from flax import nnx
import jax
import jax.numpy as jnp
from typing import Callable

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
    
    # TODO activations
    @nnx.vmap(in_axes=(None, 0), out_axes=0)
    def create_models(self, key: jax.Array):
        return MLP(hidden_dims=self.hidden_dims, rngs=nnx.Rngs(key))
    
    # def create_models(self, keys):
    #     models = []
    #     for key in keys:
    #         models.append(MLP(hidden_dims=self.hidden_dims, rngs=nnx.Rngs(key), activation=self.activation))
    #     return models
    
    # def forward(self, x):
    #     for model in self.models:
    #         x = model(x)
    #     return x 

    def __call__(self, x):
        @nnx.scan(in_axes=(0, nnx.Carry), out_axes=nnx.Carry)
        def forward(model: MLP, x):
            x = model(x)
            return x 
        z = self.activation(self.initial_linear(x))
        z = forward(self.models, z)
        z_out = self.out_layer(z)
        return z_out

if __name__ == '__main__':
    rngs = nnx.Rngs(0)
    in_dims = 10
    hidden_dims = 20
    out_dims = 5
    num_mlps = 3

    x = jax.random.normal(rngs.params(), (32, in_dims))
    model = MLPModel(in_dims, hidden_dims, out_dims, num_mlps, rngs)
    y = model(x)
    print(y.shape)  # Should be (32, out_dims)