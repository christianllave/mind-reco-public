import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
from typing import Tuple

from modules.constants import PredictionParams, DataColumns
from modules.transformation import list_agg, create_prediction_matrix, format_recommendations

def is_csr_matrix(df):
    """
    Checks if the input matrix is a (compressed) sparse matrix. If not, convert the input.
    """
    if not isinstance(df, csr_matrix):
        df = df.tocsr()

    return df

def decompose_matrix(df:csr_matrix):
    """
    Returns the number of rows, number of columns, indices, and pointers.
    """
    num_users, num_items = df.shape
    indices = df.indices
    pointers = df.indptr

    return num_users, num_items, indices, pointers

def index_users(num_users, pointers):
    """
    Returns an array of users with an item.
    Users are expressed as the index from 0 -> num_users.
    
    Note: np.any returns a matrix and can remove sparsity
    """
    
    # Generate an array range of 0 -> total number of users
    to_generate = np.arange(num_users, dtype='int32')

    # Take users where there is an item on their row. Consecutive pointers with the same value mean they're empty rows.
    return to_generate[np.ediff1d(pointers) > 0]

def get_common_matrix(m1: csr_matrix, m2: csr_matrix) -> Tuple[csr_matrix, csr_matrix]:
    """
    Get only common users and items between two csr matrices to align on sizes.
    """
    # Find minimum common shape
    common_rows = min(m1.shape[0], m2.shape[0])
    common_cols = min(m1.shape[1], m2.shape[1])
    
    # Slice both matrices to the common shape
    m1_common = m1[:common_rows, :common_cols]
    m2_common = m2[:common_rows, :common_cols]

    return m1_common, m2_common
    
def create_mask(matrix: csr_matrix) -> csr_matrix:
    """
    Create a mask of 1's for matrix data with values.
    """
    mask = matrix.copy()
    mask.data += 1
    mask.data[mask.data > 0] = 1

    return mask

def get_common_values(pred_mask:csr_matrix, valid_mask:csr_matrix) -> csr_matrix:
    """
    Creates mask of intersection (1), False Negatives (0), False Positives (-1).
    """
    # subtract pred and valid masks to see matches and mismatches
    common = pred_mask - valid_mask

    # 1 is a match, 2 is a false positive, -1 is a false negative
    common.data[common.data < 0] = 0 # remove FNs
    common.eliminate_zeros()
    
    common.data[common.data==2] = -1

    return common

def intersect_prediction_common(pred_common:csr_matrix, common:csr_matrix) -> np.array:
    """
    Return relevant predictions as positive and irrelevant predictions as negative.
    """
    # apply the common mask to pred_matrix to get the positions of relevant items
    pred_applied = common.multiply(pred_common)

    # Recover user_id (row) for each nonzero
    # np.arange(common.shape[0]) -> gets user IDs from 0 to number of users - 1
    # say N = np.diff(common.indptr) -> gets number of items in each row as an array
    # np.repeat -> a user id is repeated based on N, so if user 0 has a 3 items, we have [0,0,0]
    row_indices = np.repeat(np.arange(pred_applied.shape[0]), np.diff(pred_applied.indptr))
    col_indices = pred_applied.indices
    positions = pred_applied.data

    # Stack into (user_id, item_id) pairs
    # ex. user_ids = [0, 0, 0]
    # ex. items = [1, 2, 3]
    # then vstack -> [0, 0, 0], [1, 2, 3] -> transposed would be [0,1], [0,2], [0,3]
    return np.vstack((row_indices, col_indices, positions)).T


def compare_common_elements(pred_matrix: csr_matrix, valid_matrix: csr_matrix) -> np.array:
    """
    Compare prediction and validation sets returning the predictions as relevant (+) or irrelevant (-).
    """
    pred_common, valid_common = get_common_matrix(pred_matrix, valid_matrix)
    pred_mask = create_mask(pred_common)
    pred_mask.data+=1 # increment pred_mask by 1 so we can only have states [1, 2, -1]

    valid_mask = create_mask(valid_common)
    common = get_common_values(pred_mask, valid_mask)

    return intersect_prediction_common(pred_common, common)


def position_to_relevance(df:pd.DataFrame, rel_position:str) -> np.ndarray:
    """
    Convert the relevant position array to binary. 1 is relevant, 0 is not.
    Fulfills delta as a function the recommended item being in the set of relevant items.
    """
    return (np.stack(df[rel_position].values) > 0).astype(int)


def adjust_position(df:pd.DataFrame, rel_position:str, K=10) -> np.ndarray:
    """
    The +1 is from the original formula log2(i+1). The adjustment for the 0-index is already accounted for in the creation of positions.

    Legacy: Translate the relevant position values by +2 for an array of relevant position values from 0 to K-1.
    """
    return np.add(
        np.stack(df[rel_position].values),
        np.full(
        (len(df),K),
        np.full(
            (K,), 1)
    )
    )

def get_discounted_gain(df:pd.DataFrame, relevance:str, adjusted_position:str) -> np.ndarray:
    """
    Puts the delta function as a numerator, and the discounted gain as denominator using the adjusted position.
    """
    rel = np.stack(df[relevance].values)
    pos = np.stack(df[adjusted_position].values)
    return np.divide(rel, np.log2(pos), out=np.zeros_like(rel, dtype=float), where=pos != 1)

def dcg(df:pd.DataFrame, discounted_gain:str, as_array:bool=False) -> np.ndarray:
    """
    Sums up the discounted gains as either a single sum or cumulative sum (array).
    """
    if as_array:
        return np.cumsum(np.stack(df[discounted_gain].values), axis=1)
    else:
        return np.sum(np.stack(df[discounted_gain].values), axis=1)
    

def get_dcg(pairs, K, clean:bool=False, as_array:bool=False) -> pd.DataFrame:
    dcg_df = list_agg(pairs.drop(DataColumns.news_id_encoded, axis=1), DataColumns.user_id_encoded, DataColumns.position, DataColumns.abs_position)
    dcg_df[DataColumns.rel] = list(position_to_relevance(dcg_df, DataColumns.position))
    dcg_df[DataColumns.position] = list(adjust_position(dcg_df, DataColumns.position, K))
    dcg_df[DataColumns.dg] = list(get_discounted_gain(dcg_df, DataColumns.rel, DataColumns.position))
    dcg_df[DataColumns.dcg] = list(dcg(dcg_df, DataColumns.dg, as_array))

    if clean:
        return dcg_df[[DataColumns.user_id_encoded, DataColumns.dcg]]
    else:
        return dcg_df

def get_idcg(grouped_valid, K, clean:bool=False, as_array:bool=False) -> pd.DataFrame:
    # idcg only iterates through the minimum between the number of relevant items and K, so we "limit" using .head
    idcg_df = grouped_valid.groupby(DataColumns.user_id_encoded, as_index=False).head(K)
    idcg_df = idcg_df[[DataColumns.user_id_encoded, DataColumns.news_id_encoded]].groupby(DataColumns.user_id_encoded, as_index=False).count()

    # indicate 1 for each relevant item
    full_range = np.full(K, 1)

    # broadcast a range from 1 to K+1 to compare with each item count
    mask = np.arange(1, K + 1)[None, :] <= idcg_df[DataColumns.news_id_encoded].values[:, None]

    # create padding of 0 for array positions outside of the total number of relevant item for that user
    padded = np.where(mask, full_range, 0)

    # assign the padded array to each user
    idcg_df[DataColumns.rel] = list(padded)

    # create array of 2 to K+2 to fulfill the i+1 term
    idcg_df[DataColumns.position] = [np.arange(2, K+2)]*idcg_df.shape[0]

    # get discounted gain for each item
    idcg_df[DataColumns.dg] = list(get_discounted_gain(idcg_df, DataColumns.rel, DataColumns.position))
    idcg_df[DataColumns.idcg] = list(dcg(idcg_df, DataColumns.dg, as_array))

    if clean:
        return idcg_df[[DataColumns.user_id_encoded, DataColumns.idcg]]
    else:
        return idcg_df
    

def get_ndcg_from_parts(dcg_df, idcg_df) -> pd.DataFrame:
    """
    Use scalar values of dcg and idcg to calculate ndcg for each user.
    """
    ndcg_df = pd.merge(
        dcg_df[[DataColumns.user_id_encoded, DataColumns.dcg]],
        idcg_df[[DataColumns.user_id_encoded, DataColumns.idcg]],
        on=DataColumns.user_id_encoded,
        how='left'
        )
    ndcg_df[DataColumns.ndcg] = ndcg_df[DataColumns.dcg] / ndcg_df[DataColumns.idcg]

    return ndcg_df

def summarise_metric(df:pd.DataFrame, metric:str) -> float:
    return df[metric].mean()

def get_ndcg(pairs:pd.DataFrame, grouped_valid:pd.DataFrame, K:int) -> float:
    dcg_df = get_dcg(pairs, K, clean=True)
    idcg_df = get_idcg(grouped_valid, K, clean=True)
    ndcg_df = get_ndcg_from_parts(dcg_df, idcg_df)
    
    return summarise_metric(ndcg_df, DataColumns.ndcg)


def evaluate(train_matrix:csr_matrix, valid_matrix:csr_matrix, grouped_valid:pd.DataFrame, model, metric:str='ndcg', K:int=PredictionParams.default_k) -> float:
    if metric != 'ndcg':
        print(f'{metric} not supported')
    
    else:
        train_matrix = is_csr_matrix(train_matrix)
        test_matrix = is_csr_matrix(valid_matrix)
        num_users, _, _, pointers = decompose_matrix(test_matrix)
        relevant_users = index_users(num_users, pointers)

        # predict on relevant users in the validation set
        preds_matrix = create_prediction_matrix(relevant_users, train_matrix, model, K)

        # compare predictions with actual values in the validation matrix and return user-item pairs
        pairs = compare_common_elements(preds_matrix, test_matrix)

        # format each pair as a dataframe with assigned positions based on the recommendation order
        pairs = format_recommendations(pairs)

        return get_ndcg(pairs, grouped_valid, K)