"""
Main entry point for training models. 
"""
from flax import nnx
import jax
import jax.numpy as jnp
from typing import Callable
import optax 
from models import *
import tensorflow as tf  

def main():
    data_dict = jnp.load('./data/nn_training_data.npy', allow_pickle=True).item() # item to get dict 
    # mlp for now 
    
    pass

if __name__ == '__main__':
    main()