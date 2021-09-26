from category_encoders.cat_boost import CatBoostEncoder
from matplotlib import pyplot as plt

import typing
import pickle
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from lightgbm import LGBMRegressor

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold, StratifiedKFold
import typing
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from enum import IntEnum
from tqdm.auto import tqdm
from typing import List
from sklearn.preprocessing import RobustScaler


FOLDS = 5
THRESHOLD = 0.15
NEGATIVE_WEIGHT = 1.1
EPS = 1e-8
UNKNOWN_VALUE = 'missing'
TARGET = 'per_square_meter_price'
CATEGORICAL_STE_FEATURES = ['region', 'city', 'realty_type', 'month']
CATEGORICAL_OHE_FEATURES = []
NUM_FEATURES = [
    'floor', 'month_num', 'lat', 'lng', 'osm_amenity_points_in_0.001',
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
    'reform_mean_year_building_1000', 'reform_mean_year_building_500','total_square'
]
MODEL_PARAMS = dict(
    n_estimators=4400,
    learning_rate=0.0084858,
    lambda_l1=0.00336302,
    lambda_l2=0.0593119,
    num_leaves=589,
    min_child_samples=20,
    min_gain_to_split=0.00324483,
    feature_fraction=0.938515,
    bagging_freq=10,
    importance_type="gain",
    n_jobs=16,
    random_state=563,
    objective='regression_l1',
)


class PriceTypeEnum(IntEnum):

    OFFER_PRICE = 0
    MANUAL_PRICE = 1
    
    
def correct_floor(data):
    data['floor'] = (
        data['floor']
        .mask(data['floor'] == '-1.0', -1)
        .mask(data['floor'] == '-2.0', -2)
        .mask(data['floor'] == '-3.0', -3)
        .mask(data['floor'] == 'подвал, 1', 1)
        .mask(data['floor'] == 'подвал', -1)
        .mask(data['floor'] == 'цоколь, 1', 1)
        .mask(data['floor'] == '1,2,антресоль', 1)
        .mask(data['floor'] == 'цоколь', 0)
        .mask(data['floor'] == 'тех.этаж (6)', 6)
        .mask(data['floor'] == 'Подвал', -1)
        .mask(data['floor'] == 'Цоколь', 0)
        .mask(data['floor'] == 'фактически на уровне 1 этажа', 1)
        .mask(data['floor'] == '1,2,3', 1)
        .mask(data['floor'] == '1, подвал', 1)
        .mask(data['floor'] == '1,2,3,4', 1)
        .mask(data['floor'] == '1,2', 1)
        .mask(data['floor'] == '1,2,3,4,5', 1)
        .mask(data['floor'] == '5, мансарда', 5)
        .mask(data['floor'] == '1-й, подвал', 1)
        .mask(data['floor'] == '1, подвал, антресоль', 1)
        .mask(data['floor'] == 'мезонин', 2)
        .mask(data['floor'] == 'подвал, 1-3', 1)
        .mask(data['floor'] == '1 (Цокольный этаж)', 0)
        .mask(data['floor'] == '3, Мансарда (4 эт)', 3)
        .mask(data['floor'] == 'подвал,1', 1)
        .mask(data['floor'] == '1, антресоль', 1)
        .mask(data['floor'] == '1-3', 1)
        .mask(data['floor'] == 'мансарда (4эт)', 4)
        .mask(data['floor'] == '1, 2.', 1)
        .mask(data['floor'] == 'подвал , 1 ', 1)
        .mask(data['floor'] == '1, 2', 1)
        .mask(data['floor'] == 'подвал, 1,2,3', 1)
        .mask(data['floor'] == '1 + подвал (без отделки)', 1)
        .mask(data['floor'] == 'мансарда', 3)
        .mask(data['floor'] == '2,3', 2)
        .mask(data['floor'] == '4, 5', 4)
        .mask(data['floor'] == '1-й, 2-й', 1)
        .mask(data['floor'] == '1 этаж, подвал', 1)
        .mask(data['floor'] == '1, цоколь', 1)
        .mask(data['floor'] == 'подвал, 1-7, техэтаж', 1)
        .mask(data['floor'] == '3 (антресоль)', 3)
        .mask(data['floor'] == '1, 2, 3', 1)
        .mask(data['floor'] == 'Цоколь, 1,2(мансарда)', 1)
        .mask(data['floor'] == 'подвал, 3. 4 этаж', 3)
        .mask(data['floor'] == 'подвал, 1-4 этаж', 1)
        .mask(data['floor'] == 'подва, 1.2 этаж', 1)
        .mask(data['floor'] == '2, 3', 2)
        .mask(data['floor'] == '7,8', 7)
        .mask(data['floor'] == '1 этаж', 1)
        .mask(data['floor'] == '1-й', 1)
        .mask(data['floor'] == '3 этаж', 3)
        .mask(data['floor'] == '4 этаж', 4)
        .mask(data['floor'] == '5 этаж', 5)
        .mask(data['floor'] == 'подвал,1,2,3,4,5', 1)
        .mask(data['floor'] == 'подвал, цоколь, 1 этаж', 1)
        .mask(data['floor'] == '3, мансарда', 3)
        .mask(data['floor'] == 'цоколь, 1, 2,3,4,5,6', 1)
        .mask(data['floor'] == ' 1, 2, Антресоль', 1)
        .mask(data['floor'] == '3 этаж, мансарда (4 этаж)', 3)
        .mask(data['floor'] == 'цокольный', 0)
        .mask(data['floor'] == '1,2 ', 1)
        .mask(data['floor'] == '3,4', 3)
        .mask(data['floor'] == 'подвал, 1 и 4 этаж', 1)
        .mask(data['floor'] == '5(мансарда)', 5)
        .mask(data['floor'] == 'технический этаж,5,6', 5)
        .mask(data['floor'] == ' 1-2, подвальный', 1)
        .mask(data['floor'] == '1, 2, 3, мансардный', 1)
        .mask(data['floor'] == 'подвал, 1, 2, 3', 1)
        .mask(data['floor'] == '1,2,3, антресоль, технический этаж', 1)
        .mask(data['floor'] == '3, 4', 3)
        .mask(data['floor'] == '1-3 этажи, цоколь (188,4 кв.м), подвал (104 кв.м)', 1)
        .mask(data['floor'] == '1,2,3,4, подвал', 1)
        .mask(data['floor'] == '2-й', 2)
        .mask(data['floor'] == '1, 2 этаж', 1)
        .mask(data['floor'] == 'подвал, 1, 2', 1)
        .mask(data['floor'] == '1-7', 1)
        .mask(data['floor'] == '1 (по док-м цоколь)', 1)
        .mask(data['floor'] == '1,2,подвал ', 1)
        .mask(data['floor'] == 'подвал, 2', 2)
        .mask(data['floor'] == 'подвал,1,2,3', 1)
        .mask(data['floor'] == '1,2,3 этаж, подвал ', 1)
        .mask(data['floor'] == '1,2,3 этаж, подвал', 1)
        .mask(data['floor'] == '2, 3, 4, тех.этаж', 2)
        .mask(data['floor'] == 'цокольный, 1,2', 1)
        .mask(data['floor'] == 'Техническое подполье', -1)
        .mask(data['floor'] == '1.2', 1)
        .astype(float)
    )
    return data


def deviation_metric_one_sample(y_true: typing.Union[float, int], y_pred: typing.Union[float, int]) -> float:
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
    return np.array([deviation_metric_one_sample(y_true[n], y_pred[n]) for n in range(len(y_true))]).mean()


def median_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
    return np.median(np.abs(y_pred-y_true)/y_true)


def metrics_stat(y_true: np.array, y_pred: np.array) -> typing.Dict[str,float]:
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mdape = median_absolute_percentage_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    raif_metric = deviation_metric(y_true, y_pred)
    return {'mape':mape, 'mdape':mdape, 'rmse': rmse, 'r2': r2, 'raif_metric':raif_metric}

    
def prepare_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    fillna_cols = ['region','city','street','realty_type']
    df_new[fillna_cols] = df_new[fillna_cols].fillna(UNKNOWN_VALUE)
    return df_new


class CoeffBoostingModel():
    """
    Модель представляет из себя sklearn pipeline. Пошаговый алгоритм:
      1) в качестве обучения выбираются все данные с price_type=0
      1) все фичи делятся на три типа (numerical_features, ohe_categorical_features, ste_categorical_features):
          1.1) numerical_features - применяется StandardScaler
          1.2) ohe_categorical_featires - кодируются через one hot encoding
          1.3) ste_categorical_features - кодируются через SmoothedTargetEncoder
      2) после этого все полученные фичи конкатенируются в одно пространство фичей и подаются на вход модели Lightgbm
      3) делаем предикт на данных с price_type=1, считаем среднее отклонение реальных значений от предикта. Вычитаем это отклонение на финальном шаге (чтобы сместить отклонение к 0)

    :param numerical_features: list, список численных признаков из датафрейма
    :param ohe_categorical_features: list, список категориальных признаков для one hot encoding
    :param ste_categorical_features, list, список категориальных признаков для smoothed target encoding.
                                     Можно кодировать сразу несколько полей (например объединять категориальные признаки)
    :
    """

    def __init__(self, numerical_features: typing.List[str],
                 ohe_categorical_features: typing.List[str],
                 ste_categorical_features: typing.List[typing.Union[str, typing.List[str]]],
                 model_params: typing.Dict[str, typing.Union[str,int,float]]):
        self.num_features = numerical_features
        self.ohe_cat_features = ohe_categorical_features
        self.ste_cat_features = ste_categorical_features

        self.preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), self.num_features),
            ('ohe', OneHotEncoder(), self.ohe_cat_features),
            ('ste', CatBoostEncoder(handle_missing='value', handle_unknown='value'),
             self.ste_cat_features)])

        self.model = LGBMRegressor(**model_params)

        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', self.model)])

        self._is_fitted = False
        
        self.coeff_model = LGBMRegressor(**model_params)
        
        self.coef_preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), self.num_features+['predictions']),
            ('ohe', OneHotEncoder(), self.ohe_cat_features),
            ('ste', CatBoostEncoder(handle_missing='value', handle_unknown='value'),
             self.ste_cat_features)])
        
        self.coeff_pipeline = Pipeline(steps=[
            ('preprocessor', self.coef_preprocessor),
            ('model', self.coeff_model)])

    def _find_corr_coefficient(self, X_manual: pd.DataFrame, y_manual: pd.Series):
        """Вычисление корректирующего коэффициента

        :param X_manual: pd.DataFrame с ручными оценками
        :param y_manual: pd.Series - цены ручника
        """
        predictions = self.pipeline.predict(X_manual)
        X_manual['predictions'] = self.pipeline.predict(X_manual)
        self.coeff_pipeline.fit(X_manual, y_manual , model__feature_name=[f'{i}' for i in range(X_manual.shape[1])],
                 model__sample_weight=1/y_manual.values)
        self.__is_fitted = True

    def fit(self, X_offer: pd.DataFrame, y_offer: pd.Series,
            X_manual: pd.DataFrame, y_manual: pd.Series):
        """Обучение модели.
        ML модель обучается на данных по предложениям на рынке (цены из объявления)
        Затем вычисляется среднее отклонение между руяными оценками и предиктами для корректировки стоимости

        :param X_offer: pd.DataFrame с объявлениями
        :param y_offer: pd.Series - цена предложения (в объявлениях)
        :param X_manual: pd.DataFrame с ручными оценками
        :param y_manual: pd.Series - цены ручника
        """
        print('Fit lightgbm')
        self.pipeline.fit(X_offer, np.log1p(y_offer) , model__feature_name=[f'{i}' for i in range(X_offer.shape[1])],
                         model__sample_weight=1/np.log1p(y_offer.values)) # ,model__categorical_feature=None)
        print('Find corr coefficient')
        self._find_corr_coefficient(X_manual, np.log1p(y_manual))
        self.__is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.array:
        """Предсказание модели Предсказываем преобразованный таргет, затем конвертируем в обычную цену через обратное
        преобразование.

        :param X: pd.DataFrame
        :return: np.array, предсказания (цены на коммерческую недвижимость)
        """
        if self.__is_fitted:
            X['predictions'] = self.pipeline.predict(X)
            predictions = np.expm1(np.clip(self.coeff_pipeline.predict(X), 5,15))
            return predictions
        else:
            raise NotFittedError(
                "This {} instance is not fitted yet! Call 'fit' with appropriate arguments before predict".format(
                    type(self).__name__
                )
            )

    def save(self, path: str):
        """Сериализует модель в pickle.

        :param path: str, путь до файла
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, path: str):
        """Сериализует модель в pickle.

        :param path: str, путь до файла
        :return: Модель
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model


def main():
    
    test_path = Path('../data/test.csv')
    
    test_df = pd.read_csv(test_path)
    test_df['month'] = pd.to_datetime(test_df['date']).dt.month
    test_df['month_num'] = pd.to_datetime(test_df['date']).dt.month
    test_df = correct_floor(test_df)
    pred_df = prepare_categorical(test_df)[NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES]
    
    pred = 0

    for ifold in range(FOLDS):
        model = CoeffBoostingModel(numerical_features=NUM_FEATURES, ohe_categorical_features=CATEGORICAL_OHE_FEATURES,
                              ste_categorical_features=CATEGORICAL_STE_FEATURES, model_params=MODEL_PARAMS)
        model = model.load(f"lgbm_model_{ifold}.bin")
        y_score = model.predict(pred_df)
        pred += y_score / FOLDS
    test_sub = pd.read_csv(test_path)[['id']]
    test_sub[TARGET] = pred
    test_sub.to_csv('lgbm_submission.csv', index=False)
    
    
if __name__ == "__main__":
    main()