from dataclasses import dataclass

@dataclass(frozen=True)
class PredictionParams:
    batch_size = 1_000
    default_k = 10

@dataclass(frozen=True)
class DataColumns:
    user_id_encoded = 'user_id_encoded'
    news_id_encoded = 'news_id_encoded'
    predictions = 'predictions'
    
    position = 'position'
    abs_position = 'abs_position'

    rel = 'rel'
    dg = 'dg'
    dcg = 'dcg'
    idcg = 'idcg'
    ndcg = 'ndcg'