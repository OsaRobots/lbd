import flax 
from flax import nnx
import flax.nnx.nn as nn 
import jax
import jax.numpy as np
import optax 

class LSTM(nnx.Module):
    def __init__(self, in_dims, hidden_dims, out_dims, num_layers, rngs: nnx.Rngs):
        self.lstm_cell = nn.recurrent