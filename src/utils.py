import polars as pl
import polars.selectors as cs 
import seaborn as sns
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