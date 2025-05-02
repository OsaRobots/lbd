import polars as pl
import polars.selectors as cs 
import jax.numpy as jnp 
import jax.nn 
import numpy as np
import tensorflow as tf 
from typing import List, Tuple, Dict, Any 

def processing(df: pl.DataFrame):
    df = df.sort(pl.col('subjectID', 'trial'))

    match = cs.matches(r'^valArm(?:[1-9]|1[0-9]|20)$')
    noise_arm_match = cs.matches(r'^noiseArm(?:[1-9]|1[0-9]|20)$')

    train_df = df.filter(pl.col('phase') == 'training')

    train_df = train_df.with_columns(
        all_arm_vals = pl.concat_list(match),
        all_noise_vals = pl.concat_list(noise_arm_match),
    )

    train_df = train_df.with_columns(
        true_arm_vals = pl.struct('all_arm_vals', 'all_noise_vals')
        .map_batches(
            lambda x: x.struct.field('all_arm_vals') - x.struct.field('all_noise_vals')        
        )
    )

    train_df = train_df.with_columns(
        true_arm_rankings = pl.col('true_arm_vals')
        .map_elements(
            lambda x: x.rank(descending=True, method='min'),
            return_dtype=pl.List(pl.Int8)
        )
    )

    train_df = train_df.with_columns(
        chosenRank = pl.struct('true_arm_rankings', 'chosenArm')
        .map_batches(
            lambda x: x.struct.field('true_arm_rankings').list.get(x.struct.field('chosenArm') - 1) 
        )
    )

    grouped_chosen_ranks = train_df.group_by('trial', 'expCond').agg(pl.col('chosenRank').mean())

    return train_df, grouped_chosen_ranks

def create_input_features(df: pl.DataFrame) -> pl.DataFrame:
    """Adds state_feat1, state_feat2, and training_flag columns."""
    df = df.filter(pl.col('expCond') != 'MAB_Lin') # only people who were contextual
    df = df.fill_null(0) # fill nans with 0; only happens for test phase.

    num_arms = 20 
    state_feat_one_selector = cs.matches(r'^valArm(?:[1-9]|1[0-9]|' + str(num_arms) + r')feat1$')
    state_feat_two_selector = cs.matches(r'^valArm(?:[1-9]|1[0-9]|' + str(num_arms) + r')feat2$')

    df = df.with_columns(
        pl.when(pl.col('weight1').eq(1))
        .then(pl.concat_arr(state_feat_one_selector))
        .otherwise(pl.concat_arr(state_feat_two_selector))
        .alias('state_feat1')
    )
    df = df.with_columns(
        pl.when(pl.col('weight2').eq(2))
        .then(pl.concat_arr(state_feat_two_selector))
        .otherwise(pl.concat_arr(state_feat_one_selector))
        .alias('state_feat2')
    )

    df = df.with_columns(
        pl.when(pl.col('phase').eq('training'))
        .then(1.0) # Use float for consistency
        .otherwise(0.0)
        .alias('training_phase_flag')
    )

    df = df.sort(['subjectID', 'trial'])
    return df

def prepare_model_data(df: pl.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Prepares data lists for RNN and MLP models from the raw DataFrame.

    Returns:
        Tuple containing:
        - rnn_data: List of dicts, one per subject. Each dict contains JAX arrays
                    for full sequences (train+test) needed by the RNN, plus
                    metadata like training phase length.
        - mlp_data: List of dicts, one per subject. Each dict contains JAX arrays
                    for sequences of MLP input features and corresponding targets.
    """
    df = create_input_features(df)

    rnn_data = []
    mlp_data = []

    # default padding values for the t=0 step (no prior info)
    default_action_pad_idx = jnp.zeros(shape=(20,)) 
    default_reward_pad = 0.0
    default_flag_pad = 0.0 # no reward present before trial 1

    for subject_id, group_df in df.group_by('subjectID', maintain_order=True):
        group_df = group_df.sort('trial') # sort just to be sure

        np_state_feat1 = np.array(group_df['state_feat1'].to_list(), dtype=np.float32) # (T, num_arms)
        np_state_feat2 = np.array(group_df['state_feat2'].to_list(), dtype=np.float32) # (T, num_arms)
        np_chosen_arm_t = group_df['chosenArm'].to_numpy() # action a_t (1-indexed)
        np_reward_t = group_df['rewardObtained'].to_numpy().astype(np.float32) # reward r_t
        np_training_phase_flag_t = group_df['training_phase_flag'].to_numpy() # flag for reward r_t presentation

        # RNN preparation 

        # state features
        s_sequence = np.concatenate([np_state_feat1, np_state_feat2], axis=1) # Shape (T, 2 * num_arms)
        target_a_sequence = np_chosen_arm_t - 1 # Shape (T,), -1 to 0-index for eventual one-hot
        target_a_sequence = jax.nn.one_hot(target_a_sequence, num_classes=20)

        # roll forward to get prev features 
        a_prev_sequence = jnp.roll(target_a_sequence, shift=1, axis=0) 
        r_prev_sequence = np.roll(np_reward_t, shift=1)
        flag_prev_sequence = np.roll(np_training_phase_flag_t, shift=1)

        # pad rolled sequences
        a_prev_sequence.at[0].set(default_action_pad_idx)
        r_prev_sequence[0] = default_reward_pad
        flag_prev_sequence[0] = default_flag_pad

        # reshape for eventual concatenation. shapes will be (T, 1)
        # a_prev_sequence = a_prev_sequence[:, np.newaxis]
        r_prev_sequence = r_prev_sequence[:, np.newaxis]     
        flag_prev_sequence = flag_prev_sequence[:, np.newaxis] 

        rnn_inputs = np.concatenate([
            s_sequence,                  # (T, 2 * num_arms)
            np.asarray(a_prev_sequence), # (T, num_arms)
            r_prev_sequence,             # (T, 1)
            flag_prev_sequence           # (T, 1)
        ], axis=1)                       # Shape (T, 2 * num_arms + 2)
    

        rnn_subject_data = {
            "subjectID": subject_id,
            "rnn_inputs": rnn_inputs,              # Input s_t
            "target_actions": np.asarray(target_a_sequence),# Target a_t (indices)
        }
        rnn_data.append(rnn_subject_data)

        # MLP prep
        # use *current* np_training_phase_flag_t as context for MLP

        mlp_input_sequence = np.concatenate([
            s_sequence,                                     # (T, 2 * num_arms)
            np_training_phase_flag_t[:, np.newaxis]         # (T, 1)
        ], axis=1)                                          # Shape (T, 2 * num_arms + 1)

        mlp_subject_data = {
            "subjectID": subject_id,
            "mlp_inputs": mlp_input_sequence,    # Input features for step t
            "target_actions": np.asarray(target_a_sequence), # Target a_t (indices)
        }
        mlp_data.append(mlp_subject_data)

    return rnn_data, mlp_data

def save_training_data(df):
    rnn_data, mlp_data = prepare_model_data(df)
    np.save('./data/rnn_training_data', rnn_data)
    np.save('./data/mlp_training_data', mlp_data)

def mlp_training_data_to_tensorflow():
    mlp_data_list = list(np.load('./data/mlp_training_data.npy', allow_pickle=True))
    all_mlp_inputs = []
    all_mlp_targets = []

    for subject_data in mlp_data_list:
        mlp_inputs_np = subject_data['mlp_inputs']
        target_actions_np = subject_data['target_actions']

        all_mlp_inputs.append(mlp_inputs_np)
        all_mlp_targets.append(target_actions_np)
    
    concatenated_mlp_inputs = np.concatenate(all_mlp_inputs, axis=0)
    concatenated_mlp_targets = np.concatenate(all_mlp_targets, axis=0)

    print(f"Total MLP steps concatenated: {concatenated_mlp_inputs.shape[0]}")
    print(f"MLP input features shape: {concatenated_mlp_inputs.shape}")
    print(f"MLP target actions shape: {concatenated_mlp_targets.shape}")

    mlp_dataset = tf.data.Dataset.from_tensor_slices(
            {"inputs": concatenated_mlp_inputs, "targets": concatenated_mlp_targets}
        )
    return mlp_dataset

def rnn_training_data_to_tensorflow():
    rnn_data_list =  list(np.load('./data/rnn_training_data.npy', allow_pickle=True))
    all_rnn_inputs = []
    all_rnn_targets = []

    for subject_data in rnn_data_list:
        rnn_inputs_np = subject_data['rnn_inputs']
        target_actions_np = subject_data['target_actions']
        all_rnn_inputs.append(rnn_inputs_np)
        all_rnn_targets.append(target_actions_np)

    # reshape to get shape in form (participants, time, features) since we batch over participants 
    concatenated_rnn_inputs = np.stack(all_rnn_inputs, axis=-1).transpose(-1, 0, 1)
    concatenated_rnn_targets = np.stack(all_rnn_targets, axis=-1).transpose(-1, 0, 1)

    print(f"MLP input features shape: {concatenated_rnn_inputs.shape}")
    print(f"MLP target actions shape: {concatenated_rnn_targets.shape}")

    rnn_dataset = tf.data.Dataset.from_tensor_slices(
            {"inputs": concatenated_rnn_inputs, "targets": concatenated_rnn_targets}
        )
    return rnn_dataset

def save_tf_datasets():
    mlp_dataset = mlp_training_data_to_tensorflow()
    rnn_dataset = rnn_training_data_to_tensorflow()

    mlp_dataset.save('./data/mlp_tf_dataset')
    rnn_dataset.save('./data/rnn_tf_dataset')

def main():
    df = pl.read_csv('./data/exp1_banditData.csv', null_values='NA')
    save_training_data(df)
    save_tf_datasets()

if __name__ == '__main__':
    main()