# Notebook Outline
2. Project Context
3. Setup
   - Outputs:
       - Training data matrix
       - Validation data matrix
       - Validation truth table (user-item interactions)
5. Transform for evaluation
6. Calculate NDCG

# Project Context

## Goal:
- To have more hands-on experience working with recommendation systems outside of uni classes.
- I want to start with the basics, so for this dataset, I want to train the model just on the impression clickthrough.
- This notebook is specifically about the evaluation piece, specifically the NDCG calculation.

## Approaches:
- SGD (via fastFM)
- ALS (via Implicit / PySpark)
    - Said to handle unobserved entries easier, and converges faster than SGD.
- NMF (via Scikit-learn)
    - NMF is a batched process, so if it does work, it will be limited to batched recommendations (ex. email blasts)
- Deep Learning

Only SGD via FastFM and ALS via Implicit seem to work well on my device; however, given the theoretical benefits of ALS, I'll focus on that method for now. Meanwhile, NMF and ALS PySpark ran into memory issues. The Deep Learning implementation can be another project.

## Out of scope:
- Making use of user features and item features
    - Due to the nature of ALS. Deep learning approaches are said to handle these better.
- Using time-based features as days of week, hour of day parameters
    - Due to the nature of ALS. Deep learning approaches are said to handle these better.
- Using click frequency for relevance value (i.e. we're setting relevance as 1 regardless of click frequency)
    - For future work
- Cold-start handling

# Setup

## Imports


```python
cd ~\ds-projects\mind-reco\
```

    C:\Users\llave\ds-projects\mind-reco
    


```python
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
from typing import Tuple
```


```python
from modules.constants import PredictionParams, DataColumns
from modules.evaluation import compare_common_elements, get_ndcg, is_csr_matrix, decompose_matrix, index_users
from modules.evaluation import position_to_relevance, adjust_position, get_discounted_gain, dcg
from modules.transformation import list_agg, encode, transform_clean, format_recommendations, create_prediction_matrix
```


```python
from implicit.cpu.als import AlternatingLeastSquares as ALS
```


```python
# for presentation purposes
import warnings
warnings.filterwarnings('ignore')
```


```python
model = ALS.load('data/model.npz')
```


```python
cd ~\ds-projects\mind-reco\notebooks
```

    C:\Users\llave\ds-projects\mind-reco\notebooks
    

## Load Training and Validation interactions

The dataset combines impression and history logs between users and news articles. Each row indicates whether a user has interacted with the specific news article.


```python
grouped_train = pd.read_csv('../data/train/grouped.csv')
```


```python
grouped_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>news_id</th>
      <th>clicked</th>
      <th>user_id_encoded</th>
      <th>news_id_encoded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>U100</td>
      <td>N10121</td>
      <td>0</td>
      <td>0</td>
      <td>111</td>
    </tr>
    <tr>
      <th>1</th>
      <td>U100</td>
      <td>N10532</td>
      <td>0</td>
      <td>0</td>
      <td>470</td>
    </tr>
    <tr>
      <th>2</th>
      <td>U100</td>
      <td>N11813</td>
      <td>0</td>
      <td>0</td>
      <td>1562</td>
    </tr>
    <tr>
      <th>3</th>
      <td>U100</td>
      <td>N11856</td>
      <td>0</td>
      <td>0</td>
      <td>1602</td>
    </tr>
    <tr>
      <th>4</th>
      <td>U100</td>
      <td>N13827</td>
      <td>0</td>
      <td>0</td>
      <td>3318</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouped_valid = pd.read_csv('../data/valid/grouped.csv')
```


```python
grouped_valid.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>news_id</th>
      <th>clicked</th>
      <th>user_id_encoded</th>
      <th>news_id_encoded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>U10008</td>
      <td>N10376</td>
      <td>1</td>
      <td>4</td>
      <td>334</td>
    </tr>
    <tr>
      <th>1</th>
      <td>U10008</td>
      <td>N19990</td>
      <td>0</td>
      <td>4</td>
      <td>8741</td>
    </tr>
    <tr>
      <th>2</th>
      <td>U10008</td>
      <td>N23614</td>
      <td>1</td>
      <td>4</td>
      <td>11928</td>
    </tr>
    <tr>
      <th>3</th>
      <td>U10008</td>
      <td>N23912</td>
      <td>1</td>
      <td>4</td>
      <td>12186</td>
    </tr>
    <tr>
      <th>4</th>
      <td>U10008</td>
      <td>N29207</td>
      <td>1</td>
      <td>4</td>
      <td>16766</td>
    </tr>
  </tbody>
</table>
</div>



#### Convert to sparse matrix

Set users as rows, news articles as columns, and the number of clicks as the value.

Sparse matrices are more memory-efficient than pivot tables or dense matrices because they only store the non-zero values.


```python
from scipy.sparse import csr_matrix
```


```python
train_matrix = csr_matrix(
    (
        grouped_train['clicked'],
        (grouped_train['user_id_encoded'], grouped_train['news_id_encoded'])
    )
)
```


```python
valid_matrix = csr_matrix(
    (
        grouped_valid['clicked'],
        (grouped_valid['user_id_encoded'], grouped_valid['news_id_encoded'])
    )
)
```

# Transform for evaluation

## Get relevant users using matrices

Ensure the training and validation matrices are csr_matrix types, and decompose them into relevant parts.


```python
train_matrix = is_csr_matrix(train_matrix)
test_matrix = is_csr_matrix(valid_matrix)
num_users, num_items, indices, pointers = decompose_matrix(test_matrix)
```


```python
print(num_users, num_items)
```

    50000 51277
    

Return only relevant users (i.e. users with a corresponding item).


```python
relevant_users = index_users(num_users, pointers)
```

## Create a prediction matrix

Predict on relevant users, while holding out known interactions from training.


```python
# Set a K value
K = PredictionParams.default_k
print(K)
```

    10
    


```python
# create_prediction_matrix function in the transformation.py module
ids, _ = model.recommend(relevant_users, train_matrix[relevant_users], N=K)
k_cols = list(range(K))

preds = pd.concat([pd.DataFrame(relevant_users, columns=[DataColumns.user_id_encoded]), pd.DataFrame(ids)], axis=1)
preds[DataColumns.predictions] = list(preds[k_cols].to_numpy())
preds = preds.drop(k_cols, axis=1)
```


```python
preds.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id_encoded</th>
      <th>predictions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>[31755, 31231, 36373, 45715, 44712, 14999, 52,...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>[12522, 29888, 32114, 45303, 38824, 33218, 170...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29</td>
      <td>[31474, 26384, 31755, 3408, 31231, 51104, 3908...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>[39898, 45477, 15254, 38265, 28344, 32893, 151...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35</td>
      <td>[17995, 11244, 27408, 26384, 45715, 1907, 4487...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# assign a position 1-> K+1. Adjust index to 1 to keep the 0 values in the csr matrix
preds[DataColumns.position] = [np.arange(1, K+1)] * preds.shape[0]
```


```python
preds.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id_encoded</th>
      <th>predictions</th>
      <th>position</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>[31755, 31231, 36373, 45715, 44712, 14999, 52,...</td>
      <td>[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>[12522, 29888, 32114, 45303, 38824, 33218, 170...</td>
      <td>[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29</td>
      <td>[31474, 26384, 31755, 3408, 31231, 51104, 3908...</td>
      <td>[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>[39898, 45477, 15254, 38265, 28344, 32893, 151...</td>
      <td>[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35</td>
      <td>[17995, 11244, 27408, 26384, 45715, 1907, 4487...</td>
      <td>[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Essentially exploding each array-type column, and then combining them.

# Get the items
pred_items = preds.drop(
    DataColumns.position,
    axis=1
).explode(
    DataColumns.predictions,
    ignore_index=True
)

# Get the position of each item
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

# combine them
preds = pd.concat(
        [
            pred_items,
            pred_positions
            ],
            axis=1)

preds = preds.astype(int)
```


```python
preds.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id_encoded</th>
      <th>predictions</th>
      <th>position</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>31755</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>31231</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>36373</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>45715</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>44712</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# transform to csr_matrix
preds_matrix = csr_matrix(
    (
        preds[DataColumns.position],
        (preds[DataColumns.user_id_encoded], preds[DataColumns.predictions])
    )
)
```

## Compare predictions with actual values in the validation matrix


```python
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
```


```python
pred_common, valid_common = get_common_matrix(preds_matrix, valid_matrix)
```


```python
print(pred_common.shape, valid_common.shape)
```

    (50000, 51248) (50000, 51248)
    


```python
def create_mask(matrix: csr_matrix) -> csr_matrix:
    """
    Create a mask of 1's for matrix data with values.
    """
    mask = matrix.copy()
    mask.data += 1
    mask.data[mask.data > 0] = 1

    return mask
```


```python
pred_mask = create_mask(pred_common)
valid_mask = create_mask(valid_common)
```


```python
print(set(pred_mask.data), set(valid_mask.data))
```

    {np.int64(1)} {np.int64(1)}
    


```python
pred_mask.data+=1 # increment pred_mask by 1 so we can only have states [1, 2, -1]
```


```python
set(pred_mask.data)
```




    {np.int64(2)}




```python
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
```


```python
common = get_common_values(pred_mask, valid_mask)
```


```python
set(common.data)
```




    {np.int64(-1), np.int64(1)}




```python
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
```

Return user-item pairs marked with relevant (+) or irrelevant (-) positions and format as a dataframe.


```python
pairs = intersect_prediction_common(pred_common, common)
```


```python
pairs = pd.DataFrame(pairs, columns=[DataColumns.user_id_encoded, DataColumns.news_id_encoded, DataColumns.position])
```


```python
pairs.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id_encoded</th>
      <th>news_id_encoded</th>
      <th>position</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>52</td>
      <td>-7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5869</td>
      <td>-9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>12858</td>
      <td>-8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>14999</td>
      <td>-6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>24367</td>
      <td>-10</td>
    </tr>
  </tbody>
</table>
</div>



Assign an absolute position from the original position value.

To indicate relevance, we update the position column with 0 (if irrelevant) or the absolute position (if relevant).


```python
pairs[DataColumns.abs_position] = abs(pairs[DataColumns.position])
pairs[DataColumns.position] = pairs[DataColumns.position] * (pairs[DataColumns.position] >= 0)
```

Now we have a dataframe of recommendations to the user with the news article's position in the recommendation and its relevance.


```python
pairs.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id_encoded</th>
      <th>news_id_encoded</th>
      <th>position</th>
      <th>abs_position</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>52</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5869</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>12858</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>14999</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>24367</td>
      <td>0</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



# Calculate NDCG

NDCG mainly has two pieces: Discounted Cumulative Gain (DCG) on the numerator and Ideal DCG (IDCG) on the denominator. 

$$
\text{NDCG}(R)_K = \frac{ \sum_{i=1}^{K} \frac{\delta(i \in R)}{\log_2(i+1)} }{ \sum_{i=1}^{\min(|R|, K)} \frac{1}{\log_2(i+1)} }
$$


## Prepare DCG

Note that there is another version that uses $rel_{i}$ in place of $\delta(i \in R)$ to indicate a non-binary relevance value, which is outside the scope of this notebook.

$i$ is the position of the recommended item in the array for a certain user. $R$ is the set of relevant items, and $K$ is the number of recommendations.

$$
\text{DCG}(R)_k = \sum_{i=1}^{K} \frac{\delta(i \in R)}{\log_2(i+1)}
$$


Aggregate the items' positions as a list for each user, ordered by their absolute position, as the position value has +/- indicators.


```python
dcg_df = list_agg(pairs.drop('news_id_encoded', axis=1), 'user_id_encoded', 'position', 'abs_position')
```


```python
dcg_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id_encoded</th>
      <th>position</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
    </tr>
  </tbody>
</table>
</div>



Convert the relevant position array to binary (1 is relevant, 0 is not). This fulfills delta as a function $\delta(i \in R)$ the recommended item being in the set of relevant items.


```python
dcg_df['rel'] = list(position_to_relevance(dcg_df, 'position'))
```


```python
dcg_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id_encoded</th>
      <th>position</th>
      <th>rel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
    </tr>
  </tbody>
</table>
</div>



Adjust the position by 1 to account for ${\log_2(i+1)}$.


```python
dcg_df['pos_adj'] = list(adjust_position(dcg_df, 'position'))
```


```python
dcg_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id_encoded</th>
      <th>position</th>
      <th>rel</th>
      <th>pos_adj</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]</td>
    </tr>
  </tbody>
</table>
</div>



Puts the delta function as a numerator, and the discounted gain as denominator using the adjusted position.


```python
dcg_df['dg'] = list(get_discounted_gain(dcg_df, 'rel', 'pos_adj'))
```


```python
dcg_df['dcg'] = list(dcg(dcg_df, 'dg'))
```


```python
dcg_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id_encoded</th>
      <th>position</th>
      <th>rel</th>
      <th>pos_adj</th>
      <th>dg</th>
      <th>dcg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]</td>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]</td>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]</td>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]</td>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]</td>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## Prepare IDCG

$$
\text{IDCG}(R)_K = \sum_{i=1}^{\min(|R|, K)} \frac{1}{\log_2(i+1)}
$$


IDCG only iterates through the minimum between the number of relevant items and K, so we "limit" using .head


```python
idcg_df = grouped_valid.loc[grouped_valid['clicked']>0].groupby('user_id_encoded', as_index=False).head(K)
```

Get the number of relevant items for each user.


```python
idcg_df = idcg_df[['user_id_encoded', 'news_id_encoded']].groupby('user_id_encoded', as_index=False).count()
```


```python
idcg_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id_encoded</th>
      <th>news_id_encoded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
# preview the distribution of the number of interactions
idcg_df['news_id_encoded'].value_counts()
```




    news_id_encoded
    10    4359
    5      227
    4      195
    8      193
    9      193
    6      191
    7      182
    3      164
    1      109
    2      109
    Name: count, dtype: int64



Numpy doesn't inherently support jagged arrays, so we create a padded array. Padding the arrays with 0 doesn't interfere with succeeding calculations. This padded array accounts for the different array sizes of the $1$ term for each user.


```python
# indicate 1 for each relevant item
full_range = np.full(K, 1)

# broadcast a range from 1 to K+1 to compare with each item count
mask = np.arange(1, K + 1)[None, :] <= idcg_df['news_id_encoded'].values[:, None]

# create padding of 0 for array positions outside of the total number of relevant item for that user
padded = np.where(mask, full_range, 0)
```


```python
idcg_df['rel'] = list(padded)
```


```python
idcg_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id_encoded</th>
      <th>news_id_encoded</th>
      <th>rel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>10</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>10</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29</td>
      <td>10</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>10</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35</td>
      <td>8</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]</td>
    </tr>
  </tbody>
</table>
</div>



Assigning a position range accounts for ${\log_2(i+1)}$ and the index-1 counting.


```python
idcg_df['position'] = [np.arange(2, K+2)]*idcg_df.shape[0]
```


```python
idcg_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id_encoded</th>
      <th>news_id_encoded</th>
      <th>rel</th>
      <th>position</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>10</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]</td>
      <td>[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>10</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]</td>
      <td>[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29</td>
      <td>10</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]</td>
      <td>[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>10</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]</td>
      <td>[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35</td>
      <td>8</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]</td>
      <td>[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]</td>
    </tr>
  </tbody>
</table>
</div>



Calculate the discounted gain for each item, with the sum accounting for each user's IDCG.


```python
idcg_df['dg'] = list(get_discounted_gain(idcg_df, 'rel', 'position'))
```


```python
idcg_df['idcg'] = list(dcg(idcg_df, 'dg'))
```


```python
idcg_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id_encoded</th>
      <th>news_id_encoded</th>
      <th>rel</th>
      <th>position</th>
      <th>dg</th>
      <th>idcg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>10</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]</td>
      <td>[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]</td>
      <td>[1.0, 0.6309297535714575, 0.5, 0.4306765580733...</td>
      <td>4.543559</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>10</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]</td>
      <td>[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]</td>
      <td>[1.0, 0.6309297535714575, 0.5, 0.4306765580733...</td>
      <td>4.543559</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29</td>
      <td>10</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]</td>
      <td>[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]</td>
      <td>[1.0, 0.6309297535714575, 0.5, 0.4306765580733...</td>
      <td>4.543559</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>10</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]</td>
      <td>[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]</td>
      <td>[1.0, 0.6309297535714575, 0.5, 0.4306765580733...</td>
      <td>4.543559</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35</td>
      <td>8</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]</td>
      <td>[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]</td>
      <td>[1.0, 0.6309297535714575, 0.5, 0.4306765580733...</td>
      <td>3.953465</td>
    </tr>
  </tbody>
</table>
</div>



### Get NDCG

Combine the DCG and IDCG columns for each user into one dataframe, and divide them to get the NDCG for each user.


```python
ndcg_df = pd.merge(dcg_df[['user_id_encoded', 'dcg']],idcg_df[['user_id_encoded', 'idcg']], on='user_id_encoded', how='left')
```


```python
ndcg_df['ndcg'] = ndcg_df['dcg'] / ndcg_df['idcg']
```


```python
ndcg_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id_encoded</th>
      <th>dcg</th>
      <th>idcg</th>
      <th>ndcg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>0.0</td>
      <td>4.543559</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>0.0</td>
      <td>4.543559</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29</td>
      <td>0.0</td>
      <td>4.543559</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>0.0</td>
      <td>4.543559</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35</td>
      <td>0.0</td>
      <td>3.953465</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Take the mean across users to get the single NDCG value to assess predictions to the validation set.


```python
valid_ndcg = ndcg_df['ndcg'].mean()
```


```python
valid_ndcg
```




    np.float64(0.0009780768819512608)


