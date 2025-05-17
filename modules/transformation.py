import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.preprocessing import OrdinalEncoder

from typing import Tuple
from modules.constants import PredictionParams, DataColumns

def list_agg(df:pd.DataFrame, key:str, array_label:str, sort_index:str=None):
    """
    Converts array_label values as a list for each key in the dataframe.
    Groupby key, aggregate as list.

    Note: Cited as faster than .groupby().agg(list)
    """
    # source: https://stackoverflow.com/questions/22219004/how-to-group-dataframe-rows-into-list-in-pandas-groupby
    if sort_index is None:
        keys, values = df.sort_values(key).values.T
    else:
        df = df.sort_values([key, sort_index]).drop([sort_index], axis=1)
        keys, values = df.values.T
        
    ukeys, index = np.unique(keys, True)
    arrays = np.split(values, index[1:])
    df2 = pd.DataFrame({key:ukeys, array_label:[list(a) for a in arrays]})
    
    return df2

def transform_clean(df:pd.DataFrame, col:str, encoder:OrdinalEncoder) -> pd.DataFrame:
    return pd.concat(
        [
            df,
            encoder.transform(df[[col]]).rename(columns={col:f'{col}_encoded'}).astype(int)
            ],
        axis=1
        )

def encode(df:pd.DataFrame, col:str, encoder:OrdinalEncoder) -> Tuple[pd.DataFrame, OrdinalEncoder]:
    # I'm using handle_unknown to assign -1 for unknown values instead of giving errors.
    encoder = encoder.set_params(handle_unknown='use_encoded_value', unknown_value=-1)
    encoder.set_output(transform='pandas')
    encoder.fit(df[[col]])

    return transform_clean(df, col, encoder), encoder


def create_prediction_matrix(relevant_users:np.array, train_matrix:csr_matrix, model, K:int=10)-> pd.DataFrame:
    ids, _ = model.recommend(relevant_users, train_matrix[relevant_users], N=K)
    k_cols = list(range(K))

    preds = pd.concat([pd.DataFrame(relevant_users, columns=[DataColumns.user_id_encoded]), pd.DataFrame(ids)], axis=1)
    preds[DataColumns.predictions] = list(preds[k_cols].to_numpy())
    preds = preds.drop(k_cols, axis=1)

    # adjust index to 1 to keep the 0 values in the csr matrix
    preds[DataColumns.position] = [np.arange(1, K+1)] * preds.shape[0]

    # create user-item interactions table
    pred_items = preds.drop(DataColumns.position, axis=1).explode(DataColumns.predictions, ignore_index=True)
    pred_positions = preds.drop(
        [
            DataColumns.user_id_encoded,
            DataColumns.predictions
            ],
            axis=1
            ).explode(
                DataColumns.position,
                ignore_index=True
                )
    preds = pd.concat(
        [
            pred_items,
            pred_positions
            ],
            axis=1)
    preds = preds.astype(int)

    # transform to csr_matrix
    preds_matrix = csr_matrix(
        (
            preds[DataColumns.position],
            (preds[DataColumns.user_id_encoded], preds[DataColumns.predictions])
        )
    )

    return preds_matrix


def format_recommendations(pairs:np.array) -> pd.DataFrame:
    """
    Assign position values to each user-item pair.
    """
    pairs = pd.DataFrame(pairs, columns=[DataColumns.user_id_encoded, DataColumns.news_id_encoded, DataColumns.position])
    pairs[DataColumns.abs_position] = abs(pairs[DataColumns.position])
    pairs[DataColumns.position] = pairs[DataColumns.position] * (pairs[DataColumns.position] >= 0)

    return pairs

