import polars as pl
import polars.selectors as cs 
import jax.numpy as jnp 
import jax.nn 
import numpy as np
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

    # Define default padding values for the t=0 step (no prior info)
    # Action indices are 1-based in CSV, will convert to 0-based. Pad with -1? Or use 0 and reserve it? Let's use 0, assuming no arm index 0.
    default_action_pad_idx = 0
    default_reward_pad = 0.0
    default_flag_pad = 0.0 # Indicates no reward present before trial 1

    for subject_id, group_df in df.group_by('subjectID', maintain_order=True):
        seq_len = len(group_df)
        if seq_len == 0:
            continue

        # --- Convert relevant columns to NumPy arrays ---
        # Using NumPy for intermediate steps like np.roll
        np_state_feat1 = np.array(group_df['state_feat1'].to_list(), dtype=np.float32) # (T, num_arms)
        np_state_feat2 = np.array(group_df['state_feat2'].to_list(), dtype=np.float32) # (T, num_arms)
        np_chosen_arm_t = group_df['chosenArm'].to_numpy() # Action a_t (1-based index)
        np_reward_t = group_df['rewardObtained'].to_numpy().astype(np.float32) # Reward r_t
        np_reward_present_flag_t = group_df['training_phase_flag'].to_numpy() # Flag for reward r_t presentation

        # --- 1. Prepare RNN Data ---

        # State s_t: Concatenate features for all arms at step t
        s_sequence = np.concatenate([np_state_feat1, np_state_feat2], axis=1) # Shape (T, 2 * num_arms)

        # Target Action a_t (convert to 0-based index)
        target_a_sequence = np_chosen_arm_t - 1 # Shape (T,)

        # Previous Action a_{t-1} (0-based index, padded)
        a_prev_sequence = np.roll(target_a_sequence, shift=1)
        a_prev_sequence[0] = default_action_pad_idx # Pad first step

        # Previous Reward r_{t-1} (padded)
        r_prev_sequence = np.roll(np_reward_t, shift=1)
        r_prev_sequence[0] = default_reward_pad

        # Previous Reward Present Flag reward_present_{t-1} (padded)
        flag_prev_sequence = np.roll(np_reward_present_flag_t, shift=1)
        flag_prev_sequence[0] = default_flag_pad

        # Add a channel dimension for scalar inputs (reward, flag) for convention
        r_prev_sequence = r_prev_sequence[:, np.newaxis]     # Shape (T, 1)
        flag_prev_sequence = flag_prev_sequence[:, np.newaxis] # Shape (T, 1)

        # Metadata: Training phase length
        # np_reward_present_flag_t is 1 for training steps
        train_phase_len = int(np.sum(np_reward_present_flag_t))

        # Store subject's data (convert final arrays to JAX)
        rnn_subject_data = {
            "subjectID": subject_id,
            "states": jnp.array(s_sequence),              # Input s_t
            "prev_actions": jnp.array(a_prev_sequence),   # Input a_{t-1} (indices)
            "prev_rewards": jnp.array(r_prev_sequence),   # Input r_{t-1}
            "prev_reward_flags": jnp.array(flag_prev_sequence), # Input flag_{t-1}
            "target_actions": jnp.array(target_a_sequence),# Target a_t (indices)
            "train_phase_length": train_phase_len,
            "seq_length": seq_len,
        }
        rnn_data.append(rnn_subject_data)

        # --- 2. Prepare MLP Data ---

        # Input features at step t: s_t features + reward_present_flag_t
        # We use reward_present_flag_t (flag for *current* step) as context for MLP
        mlp_input_sequence = np.concatenate([
            s_sequence,                                     # (T, 2 * num_arms)
            np_reward_present_flag_t[:, np.newaxis]         # (T, 1)
        ], axis=1)                                          # Shape (T, 2 * num_arms + 1)

        mlp_subject_data = {
            "subjectID": subject_id,
            "mlp_inputs": jnp.array(mlp_input_sequence),    # Input features for step t
            "target_actions": jnp.array(target_a_sequence), # Target a_t (indices)
            "seq_length": seq_len,
        }
        mlp_data.append(mlp_subject_data)

    return rnn_data, mlp_data

def learning_setup(df: pl.DataFrame):
    # cs.matches(r'^valArm(?:[1-9]|1[0-9]|20)feat(?:[1-2])$')
    # state feats regex

    state_feat_one_selector = cs.matches(r'^valArm(?:[1-9]|1[0-9]|20)feat1$')
    state_feat_two_selector = cs.matches(r'^valArm(?:[1-9]|1[0-9]|20)feat2$')

    # state feat1 features are features which have weight1 = 1 attached to them
    participant_level_features = {}
    df = df.with_columns(
        pl.when(pl.col('weight1').eq(1))
        .then(pl.concat_arr(state_feat_one_selector))
        .otherwise(pl.concat_arr(state_feat_two_selector)) # weight1 \ne 1, get the state_feat_two_selector vals, since those correspond to weight1 
        .alias('state_feat1')
    )   

    # state feat2 features are features which have weight2 = 2 attached to them
    df = df.with_columns(
        pl.when(pl.col('weight2').eq(2))
        .then(pl.concat_arr(state_feat_two_selector))
        .otherwise(pl.concat_arr(state_feat_one_selector)) # weight2 \ne 2 --> weight1 \eq 2 --> get the state_feat_one_selector vals, since those correspond to weight = 2 
        .alias('state_feat2')
    )

    df = df.with_columns(
        pl.when(pl.col('phase').eq('training'))
        .then(1)
        .otherwise(0)
        .alias('training_flag')
    )

    unique_subject_ids = df.select(pl.col('subjectID')).unique()
    for unique_subject_id in unique_subject_ids.rows(named=True):

        df_subset = df.filter(pl.col('subjectID') == unique_subject_id['subjectID'])
        df_subset_train_phase = df_subset.filter(pl.col('phase') == 'training').sort(pl.col('trial'))
        df_subset_test_phase = df_subset.filter(pl.col('phase') == 'test').sort(pl.col('trial'))
        
        state_feat_one_train = df_subset_train_phase.select(pl.col('state_feat1')).to_jax()
        state_feat_two_train = df_subset_train_phase.select(pl.col('state_feat2')).to_jax()
        arm_train = jax.nn.one_hot(df_subset_train_phase.select(pl.col('chosenArm')).to_jax() - 1, 20).squeeze(1)
        reward_train = df_subset_train_phase.select(pl.col('rewardObtained')).to_jax()

        state_feat_one_test = df_subset_test_phase.select(pl.col('state_feat1')).to_jax()
        state_feat_two_test = df_subset_test_phase.select(pl.col('state_feat2')).to_jax()
        arm_test = jax.nn.one_hot(df_subset_test_phase.select(pl.col('chosenArm')).to_jax() - 1, 20).squeeze(1)
        reward_test = df_subset_test_phase.select(pl.col('rewardObtained')).to_jax()

        feature_dict_train = {
            'state_feat1': state_feat_one_train,
            'state_feat2': state_feat_two_train,
            'chosenArm': arm_train,
            'rewardObtained': reward_train,
        }

        feature_dict_test = {
            'state_feat1': state_feat_one_test,
            'state_feat2': state_feat_two_test,
            'chosenArm': arm_test,
            'rewardObtained': reward_test,
        }

        feature_dict = {'training_phase': feature_dict_train,
                        'test_phase': feature_dict_test,
        }

        participant_level_features[unique_subject_id['subjectID']] = feature_dict

    return participant_level_features

def save_training_data(feature_dict):
    jnp.save('./data/nn_training_data', feature_dict)

def main():
    # df = pl.read_csv('./data/exp1_banditData.csv', null_values='NA')
    # participant_level_features = learning_setup(df)
    df = pl.read_csv('./data/exp1_banditData.csv', null_values='NA')
    rnn_data, mlp_data = prepare_model_data(df)

    if rnn_data:
        print("\n--- Example RNN Data (Subject 0) ---")
        first_subject_rnn = rnn_data[0]
        print(f"Subject ID: {first_subject_rnn['subjectID']}")
        print(f"States shape: {first_subject_rnn['states'].shape}")
        print(f"Prev Actions shape: {first_subject_rnn['prev_actions'].shape}")
        print(f"Prev Rewards shape: {first_subject_rnn['prev_rewards'].shape}")
        print(f"Prev Flags shape: {first_subject_rnn['prev_reward_flags'].shape}")
        print(f"Target Actions shape: {first_subject_rnn['target_actions'].shape}")
        print(f"Train Phase Length: {first_subject_rnn['train_phase_length']}")
        print(f"Total Sequence Length: {first_subject_rnn['seq_length']}")

    # Example: Accessing data for the first subject for MLP
    if mlp_data:
        print("\n--- Example MLP Data (Subject 0) ---")
        first_subject_mlp = mlp_data[0]
        print(f"Subject ID: {first_subject_mlp['subjectID']}")
        print(f"MLP Inputs shape: {first_subject_mlp['mlp_inputs'].shape}")
        print(f"Target Actions shape: {first_subject_mlp['target_actions'].shape}")
        print(f"Total Sequence Length: {first_subject_mlp['seq_length']}")
        
    # one_hots = jax.nn.one_hot(feature_dict['chosenArm'] - 1, 20).squeeze(1)
    # print(one_hots.mean(axis=0))
    # save_training_data(participant_level_features)
if __name__ == '__main__':
    main()