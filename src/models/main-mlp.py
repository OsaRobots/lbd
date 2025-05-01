from flax import nnx
import jax
import jax.numpy as jnp
import pandas as pd 
import numpy as np
import tensorflow as tf 
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
    
    def fit(self, data):
        """
        We train the MLP model such that it takes 
        """



if __name__ == '__main__':
    # rngs = nnx.Rngs(0)
    # in_dims = 10
    # hidden_dims = 20
    # out_dims = 5
    # num_mlps = 3

    # x = jax.random.normal(rngs.params(), (32, in_dims))
    # model = MLPModel(in_dims, hidden_dims, out_dims, num_mlps, rngs)
    # y = model(x)
    # print(y.shape)  # Should be (32, out_dims)
    mlp_data_list = list(jnp.load('./data/mlp_training_data.npy', allow_pickle=True))
    # dataset = tf.data.Dataset.from_tensor_slices(pd.DataFrame.from_dict(data).to_dict(orient="list"))

    all_mlp_inputs = []
    all_mlp_targets = []

    for subject_data in mlp_data_list:
        # Ensure arrays are numpy arrays (they might be JAX arrays if saved directly)
        mlp_inputs_np = np.asarray(subject_data['mlp_inputs'])
        target_actions_np = np.asarray(subject_data['target_actions'])

        # Basic validation
        if mlp_inputs_np.shape[0] != target_actions_np.shape[0]:
            print(f"Warning: Mismatch in sequence length for subject {subject_data.get('subjectID', 'Unknown')}. Skipping.")
            continue
        if mlp_inputs_np.ndim != 2 or target_actions_np.ndim != 1:
            print(f"Warning: Unexpected array dimensions for subject {subject_data.get('subjectID', 'Unknown')}. Skipping.")
            continue

        all_mlp_inputs.append(mlp_inputs_np)
        all_mlp_targets.append(target_actions_np)

    if not all_mlp_inputs:
        print("No valid data found after checking subjects.")
        # exit()
    
    concatenated_mlp_inputs = np.concatenate(all_mlp_inputs, axis=0)
    concatenated_mlp_targets = np.concatenate(all_mlp_targets, axis=0)

    print(f"Total MLP steps concatenated: {concatenated_mlp_inputs.shape[0]}")
    print(f"MLP input features shape: {concatenated_mlp_inputs.shape}")
    print(f"MLP target actions shape: {concatenated_mlp_targets.shape}")

    try:
        mlp_dataset = tf.data.Dataset.from_tensor_slices(
            (concatenated_mlp_inputs, concatenated_mlp_targets)
        )

        # --- Optional: Shuffle, Batch, Prefetch ---
        total_steps = concatenated_mlp_inputs.shape[0]
        # Adjust buffer size based on your memory constraints
        shuffle_buffer_size = min(total_steps, 10000)

        mlp_dataset = mlp_dataset.shuffle(buffer_size=shuffle_buffer_size)
        mlp_dataset = mlp_dataset.batch(32)
        mlp_dataset = mlp_dataset.prefetch(tf.data.AUTOTUNE)

        print("\nTensorFlow Dataset for MLP created successfully!")
        # You can now iterate over mlp_dataset in your training loop
        # for batch_inputs, batch_targets in mlp_dataset.take(1):
        #     print("Example Batch Shapes:")
        #     print("Inputs:", batch_inputs.shape)
        #     print("Targets:", batch_targets.shape)

    except Exception as e:
        print(f"Error creating TensorFlow dataset: {e}")
        print(f"Input dtypes: {concatenated_mlp_inputs.dtype}, {concatenated_mlp_targets.dtype}")
