import pandas as pd
import numpy as np
import json
import re

import typing
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor

import torch
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts





train = pd.read_csv('../data/train.csv', low_memory=False)
test = pd.read_csv('../data/test.csv')

with open('../data/russian-cities.json', 'r') as json_file:
    cities_data = json.load(json_file)
population_dict = {city_dict['name']: city_dict['population'] for city_dict in cities_data}
train['population'] = train['city'].map(population_dict)
test['population'] = test['city'].map(population_dict)

pattern = r'[a-zа-я\s]'

train['floor_count_comma'] = train['floor'].str.count(',').fillna(0)
test['floor_count_comma'] = test['floor'].str.count(',').fillna(0)

train['floor']=train['floor'].fillna('missing').str.lower()
test['floor']=test['floor'].fillna('missing').str.lower()
train['floor']=train['floor'].str.replace(' ','')
test['floor']=test['floor'].str.replace(' ','')
train['floor']=train['floor'].str.replace('.0','', regex=False)
test['floor']=test['floor'].str.replace('.0','', regex=False)

train['floor_under'] = (train['floor'].str.count('подвал')  > 0).astype(int)
train['floor_upper'] = (train['floor'].str.count('мансарда') > 0).astype(int)
train['floor_ground'] = (train['floor'].str.count('цоколь') > 0).astype(int)
test['floor_under'] = (test['floor'].str.count('подвал')  > 0).astype(int)
test['floor_upper'] = (test['floor'].str.count('мансарда') > 0).astype(int)
test['floor_ground'] = (test['floor'].str.count('цоколь') > 0).astype(int)

def min_floor(s):
    x = 0
    try:
        x = int(s)
    except:
        pass
    if x == 0:
        try:
            ss = re.sub(pattern, '', s)
            if ss[0] == ',':
                ss = ss[1:]
            x = int(ss.split(',')[0])
        except:
            pass
    return x

def max_floor(s):
    x = 0
    try:
        x = int(s)
    except:
        pass
    if x == 0:
        try:
            ss = re.sub(pattern, '', s)
            if ss[-1] == ',':
                ss = ss[1:]
            x = int(ss.split(',')[-1])
        except:
            pass
    return x

train['num_min_floor'] = train['floor'].apply(min_floor).fillna(0)
test['num_min_floor'] = test['floor'].apply(min_floor).fillna(0)
train['num_max_floor'] = train['floor'].apply(max_floor).fillna(0)
test['num_max_floor'] = test['floor'].apply(max_floor).fillna(0)

TARGET = 'per_square_meter_price'
# признаки (или набор признаков), для которых применяем smoothed target encoding
CATEGORICAL_FEATURES = ['region', 'city', 'realty_type', 'street','floor','osm_city_nearest_name']

# численные признаки
NUM_FEATURES = ['lat', 'lng', 'osm_amenity_points_in_0.001',
       'osm_amenity_points_in_0.005', 'osm_amenity_points_in_0.0075',
       'osm_amenity_points_in_0.01', 'osm_building_points_in_0.001',
       'osm_building_points_in_0.005', 'osm_building_points_in_0.0075',
       'osm_building_points_in_0.01', 'osm_catering_points_in_0.001',
       'osm_catering_points_in_0.005', 'osm_catering_points_in_0.0075',
       'osm_catering_points_in_0.01', 'osm_city_closest_dist',
       'osm_city_nearest_population',
       'osm_crossing_closest_dist', 'osm_crossing_points_in_0.001',
       'osm_crossing_points_in_0.005', 'osm_crossing_points_in_0.0075',
       'osm_crossing_points_in_0.01', 'osm_culture_points_in_0.001',
       'osm_culture_points_in_0.005', 'osm_culture_points_in_0.0075',
       'osm_culture_points_in_0.01', 'osm_finance_points_in_0.001',
       'osm_finance_points_in_0.005', 'osm_finance_points_in_0.0075',
       'osm_finance_points_in_0.01', 'osm_healthcare_points_in_0.005',
       'osm_healthcare_points_in_0.0075', 'osm_healthcare_points_in_0.01',
       'osm_historic_points_in_0.005', 'osm_historic_points_in_0.0075',
       'osm_historic_points_in_0.01', 'osm_hotels_points_in_0.005',
       'osm_hotels_points_in_0.0075', 'osm_hotels_points_in_0.01',
       'osm_leisure_points_in_0.005', 'osm_leisure_points_in_0.0075',
       'osm_leisure_points_in_0.01', 'osm_offices_points_in_0.001',
       'osm_offices_points_in_0.005', 'osm_offices_points_in_0.0075',
       'osm_offices_points_in_0.01', 'osm_shops_points_in_0.001',
       'osm_shops_points_in_0.005', 'osm_shops_points_in_0.0075',
       'osm_shops_points_in_0.01', 'osm_subway_closest_dist',
       'osm_train_stop_closest_dist', 'osm_train_stop_points_in_0.005',
       'osm_train_stop_points_in_0.0075', 'osm_train_stop_points_in_0.01',
       'osm_transport_stop_closest_dist', 'osm_transport_stop_points_in_0.005',
       'osm_transport_stop_points_in_0.0075',
       'osm_transport_stop_points_in_0.01',
       'reform_count_of_houses_1000', 'reform_count_of_houses_500',
       'reform_house_population_1000', 'reform_house_population_500',
       'reform_mean_floor_count_1000', 'reform_mean_floor_count_500',
       'reform_mean_year_building_1000', 'reform_mean_year_building_500', 'total_square']

import pandas as pd
UNKNOWN_VALUE = 'missing'

def prepare_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет пропущенные категориальные переменные
    :param df: dataframe, обучающая выборка
    :return: dataframe
    """
    df_new = df.copy()
    fillna_cols = list(CATEGORICAL_FEATURES)
    df_new[fillna_cols] = df_new[fillna_cols].fillna(UNKNOWN_VALUE)
    return df_new
train = prepare_categorical(train)
test = prepare_categorical(test)

# в трейне те значения, которые встречаются только один раз заменяем на missing
for c in CATEGORICAL_FEATURES:
    q = train[c].value_counts()
    single = q.loc[q.values==1].index.tolist()
    changed_in_train = train.loc[train[c].isin(single),c].shape[0]
    changed_in_test = test.loc[train[c].isin(single),c].shape[0]
    train.loc[train[c].isin(single),c] = UNKNOWN_VALUE
    test.loc[test[c].isin(single),c] = UNKNOWN_VALUE
    print('category',c,'changed in train',changed_in_train,'changed in test',changed_in_test)

#убираем из теста те значения, которых нет в трейн
for c in CATEGORICAL_FEATURES:
    train_uniq = train[c].unique()
    count_changed = 0
    for v in test[c].unique():
        if v not in train_uniq:
            count_changed += test.loc[test[c] == v, c].shape[0]
            test.loc[test[c] == v, c] = UNKNOWN_VALUE
        if count_changed > 0:
            print('changed in category ', c, 'value', v, count_changed, 'times')
            
train['date'] = pd.to_datetime(train['date'])
train['month'] = train['date'].dt.month
test['date'] = pd.to_datetime(test['date'])
test['month'] = test['date'].dt.month


THRESHOLD = 0.15
NEGATIVE_WEIGHT = 1.1


def deviation_metric_one_sample(y_true: typing.Union[float, int], y_pred: typing.Union[float, int]) -> float:
    """
    Реализация кастомной метрики для хакатона.

    :param y_true: float, реальная цена
    :param y_pred: float, предсказанная цена
    :return: float, значение метрики
    """
    deviation = (y_pred - y_true) / np.maximum(1e-8, y_true)
    if np.abs(deviation) <= THRESHOLD:
        return 0
    elif deviation <= - 4 * THRESHOLD:
        return 9 * NEGATIVE_WEIGHT
    elif deviation < -THRESHOLD:
        return NEGATIVE_WEIGHT * ((deviation / THRESHOLD) + 1) ** 2
    elif deviation < 4 * THRESHOLD:
        return ((deviation / THRESHOLD) - 1) ** 2
    else:
        return 9


def deviation_metric(y_true: np.array, y_pred: np.array) -> float:
    return np.mean([deviation_metric_one_sample(y_true[n], y_pred[n]) for n in range(len(y_true))]) #.mean()

def median_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
    return np.median(np.abs(y_pred-y_true)/y_true)

def metrics_stat(y_true: np.array, y_pred: np.array) -> typing.Dict[str,float]:
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mdape = median_absolute_percentage_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    raif_metric = deviation_metric(y_true, y_pred)
    return {'mape':mape, 'mdape':mdape, 'rmse': rmse, 'r2': r2, 'raif_metric':raif_metric}

train_df = train.drop(columns=['id','per_square_meter_price','date'])
test_df = test.drop(columns=['id','date'])[train_df.columns.tolist()]


dim_size_ = []
for c in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    le.fit(train_df[c].values.tolist() + test_df[c].values.tolist())
    train_df[c] = le.transform(train_df[c])
    test_df[c] = le.transform(test_df[c])
    dim_size_.append(len(np.unique(train_df[c].values.tolist() + test_df[c].values.tolist())))
    
cat_idxs = [train_df.columns.tolist().index(x) for x in CATEGORICAL_FEATURES]

for c in train_df.columns:
    median_val = train_df[c].median()
    train_df[c] = train_df[c].fillna(median_val)
    test_df[c] = test_df[c].fillna(median_val)
    
kf = KFold(n_splits=5, shuffle=True, random_state=239)

class DevMetric(Metric):
    def __init__(self):
        self._name = "DevMetric"
        self._maximize = False

    def __call__(self, y_true, y_score):
        #print(y_true.shape, y_score.shape, y_true.dtype)
        return metrics_stat(np.expm1(y_true.flatten()), 
                            np.expm1(np.clip(y_score.flatten(),5,15))
                           )['raif_metric']
    
def MAPELoss(y_pred, y_true):
    return torch.mean(torch.abs(y_true - y_pred) / y_true).clone()
    
tabnet_params = dict(
    cat_idxs=cat_idxs,
    cat_dims=dim_size_,
    cat_emb_dim=5,
    n_d = 8,
    n_a = 8,
    n_steps = 1,
    gamma = 5,
    n_independent = 2,
    n_shared = 2,
    lambda_sparse = 0,
    optimizer_fn = AdamW,
    optimizer_params = dict(lr = (1e-2), weight_decay=0.0),
    mask_type = "entmax",
    scheduler_params = dict(T_0=120, T_mult=1, eta_min=1e-5, last_epoch=-1, verbose=False),
    scheduler_fn = CosineAnnealingWarmRestarts,
    seed = 42,
    verbose = 10
)

ifold = 0
for tr,va in kf.split(train_df):
    df_tr = train_df.loc[tr].reset_index(drop=True).values
    df_va = train_df.loc[va].reset_index(drop=True).values
    tr_y = np.log1p(train.loc[tr,[TARGET]].values)
    va_y = np.log1p(train.loc[va,[TARGET]].values)
    
    va_y = va_y[train_df.loc[va].price_type.values == 1]
    df_va = df_va[train_df.loc[va].price_type.values == 1]
    
    c_1_tr_y = tr_y[train_df.loc[tr].price_type.values == 1]
    c_1_df_tr = df_tr[train_df.loc[tr].price_type.values == 1]
    
    clf = TabNetRegressor(**tabnet_params)
    clf.fit(
      df_tr, tr_y,
      eval_set=[(c_1_df_tr, c_1_tr_y), (df_va, va_y)],
      max_epochs = 120,
      patience = 15,
      batch_size = 256, 
      virtual_batch_size = 256,
      num_workers = 4,
      drop_last = False,
      eval_metric=[DevMetric],
      loss_fn=MAPELoss
    )
    clf.save_model('model_p1_'+str(ifold)+'.pth')
    ifold += 1