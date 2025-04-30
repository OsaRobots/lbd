import polars as pl
import polars.selectors as cs 
from inspect import signature
from typing import Callable

def list_eval_ref(
    listcol: pl.Expr | str,
    op: Callable[..., pl.Expr],
    *ref_cols: str | pl.Expr,
):
    if len(ref_cols)==0:
        ref_cols = tuple([x for x in signature(op).parameters.keys()][1:])
    
    args_to_op = [pl.element().struct[0].explode()] + [
        pl.element().struct[i + 1] for i in range(len(ref_cols))
    ]
    return pl.concat_list(pl.struct(listcol, *ref_cols)).list.eval(op(*args_to_op))

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

def learning_setup(df: pl.DataFrame):
    # cs.matches(r'^valArm(?:[1-9]|1[0-9]|20)feat(?:[1-2])$')
    # state feats regex
    state_feat_one_selector = cs.matches(r'^valArm(?:[1-9]|1[0-9]|20)feat1$')
    state_feat_two_selector = cs.matches(r'^valArm(?:[1-9]|1[0-9]|20)feat2$')

    # state feat1 features are features which have weight1 = 1 attached to them
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

    state_feat_one = df.select(pl.col('state_feat1')).to_jax()
    state_feat_two = df.select(pl.col('state_feat2')).to_jax()
    arm = df.select(pl.col('chosenArm')).to_jax()
    reward = df.select(pl.col('rewardObtained')).to_jax()
    training_flag = df.select(pl.col('training_flag')).to_jax()
    
    feature_dict = {
        'state_feat1': state_feat_one,
        'state_feat2': state_feat_two,
        'chosenArm': arm,
        'rewardObtained': reward,
        'training_flag': training_flag
    }

    return feature_dict
    # line up so that weight1 is always 1 and weight 2 is always 2  
    


if __name__ == '__main__':
    df = pl.read_csv('./data/exp1_banditData.csv', null_values='NA')
    feature_dict = learning_setup(df)
    print(feature_dict['state_feat1'].shape)
    #train_df, grouped_chosen_ranks = processing(df)