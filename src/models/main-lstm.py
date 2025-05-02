import flax 
from flax import nnx
import flax.nnx.nn as nn 
import jax
import jax.numpy as np
import optax 

class LSTMModel(nnx.Module):
    def __init__(self, in_dims, hidden_dims, out_dims, num_layers, rngs: nnx.Rngs):
        if num_layers == 1:
            self.lstm_cell = nnx.OptimizedLSTMCell(in_features=in_dims, hidden_features=hidden_dims)