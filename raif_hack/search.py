import os
import pickle
import typing
from dataclasses import dataclass
from enum import IntEnum

import hydra
import numpy as np
import pandas as pd
import sqlalchemy
from category_encoders.cat_boost import CatBoostEncoder
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
from lightgbm import LGBMRegressor
from optuna import Trial, create_study
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from raif_hack.metrics import metrics_stat
from raif_hack.settings import CATEGORICAL_OHE_FEATURES, CATEGORICAL_STE_FEATURES, NUM_FEATURES, TARGET
from raif_hack.settings import TRAIN_PATH as train_path

FOLDS = 5


def correct_floor(data):
    data["floor"] = (
        data["floor"]
        .mask(data["floor"] == "-1.0", -1)
        .mask(data["floor"] == "-2.0", -2)
        .mask(data["floor"] == "-3.0", -3)
        .mask(data["floor"] == "подвал, 1", 1)
        .mask(data["floor"] == "подвал", -1)
        .mask(data["floor"] == "цоколь, 1", 1)
        .mask(data["floor"] == "1,2,антресоль", 1)
        .mask(data["floor"] == "цоколь", 0)
        .mask(data["floor"] == "тех.этаж (6)", 6)
        .mask(data["floor"] == "Подвал", -1)
        .mask(data["floor"] == "Цоколь", 0)
        .mask(data["floor"] == "фактически на уровне 1 этажа", 1)
        .mask(data["floor"] == "1,2,3", 1)
        .mask(data["floor"] == "1, подвал", 1)
        .mask(data["floor"] == "1,2,3,4", 1)
        .mask(data["floor"] == "1,2", 1)
        .mask(data["floor"] == "1,2,3,4,5", 1)
        .mask(data["floor"] == "5, мансарда", 5)
        .mask(data["floor"] == "1-й, подвал", 1)
        .mask(data["floor"] == "1, подвал, антресоль", 1)
        .mask(data["floor"] == "мезонин", 2)
        .mask(data["floor"] == "подвал, 1-3", 1)
        .mask(data["floor"] == "1 (Цокольный этаж)", 0)
        .mask(data["floor"] == "3, Мансарда (4 эт)", 3)
        .mask(data["floor"] == "подвал,1", 1)
        .mask(data["floor"] == "1, антресоль", 1)
        .mask(data["floor"] == "1-3", 1)
        .mask(data["floor"] == "мансарда (4эт)", 4)
        .mask(data["floor"] == "1, 2.", 1)
        .mask(data["floor"] == "подвал , 1 ", 1)
        .mask(data["floor"] == "1, 2", 1)
        .mask(data["floor"] == "подвал, 1,2,3", 1)
        .mask(data["floor"] == "1 + подвал (без отделки)", 1)
        .mask(data["floor"] == "мансарда", 3)
        .mask(data["floor"] == "2,3", 2)
        .mask(data["floor"] == "4, 5", 4)
        .mask(data["floor"] == "1-й, 2-й", 1)
        .mask(data["floor"] == "1 этаж, подвал", 1)
        .mask(data["floor"] == "1, цоколь", 1)
        .mask(data["floor"] == "подвал, 1-7, техэтаж", 1)
        .mask(data["floor"] == "3 (антресоль)", 3)
        .mask(data["floor"] == "1, 2, 3", 1)
        .mask(data["floor"] == "Цоколь, 1,2(мансарда)", 1)
        .mask(data["floor"] == "подвал, 3. 4 этаж", 3)
        .mask(data["floor"] == "подвал, 1-4 этаж", 1)
        .mask(data["floor"] == "подва, 1.2 этаж", 1)
        .mask(data["floor"] == "2, 3", 2)
        .mask(data["floor"] == "7,8", 7)
        .mask(data["floor"] == "1 этаж", 1)
        .mask(data["floor"] == "1-й", 1)
        .mask(data["floor"] == "3 этаж", 3)
        .mask(data["floor"] == "4 этаж", 4)
        .mask(data["floor"] == "5 этаж", 5)
        .mask(data["floor"] == "подвал,1,2,3,4,5", 1)
        .mask(data["floor"] == "подвал, цоколь, 1 этаж", 1)
        .mask(data["floor"] == "3, мансарда", 3)
        .mask(data["floor"] == "цоколь, 1, 2,3,4,5,6", 1)
        .mask(data["floor"] == " 1, 2, Антресоль", 1)
        .mask(data["floor"] == "3 этаж, мансарда (4 этаж)", 3)
        .mask(data["floor"] == "цокольный", 0)
        .mask(data["floor"] == "1,2 ", 1)
        .mask(data["floor"] == "3,4", 3)
        .mask(data["floor"] == "подвал, 1 и 4 этаж", 1)
        .mask(data["floor"] == "5(мансарда)", 5)
        .mask(data["floor"] == "технический этаж,5,6", 5)
        .mask(data["floor"] == " 1-2, подвальный", 1)
        .mask(data["floor"] == "1, 2, 3, мансардный", 1)
        .mask(data["floor"] == "подвал, 1, 2, 3", 1)
        .mask(data["floor"] == "1,2,3, антресоль, технический этаж", 1)
        .mask(data["floor"] == "3, 4", 3)
        .mask(data["floor"] == "1-3 этажи, цоколь (188,4 кв.м), подвал (104 кв.м)", 1)
        .mask(data["floor"] == "1,2,3,4, подвал", 1)
        .mask(data["floor"] == "2-й", 2)
        .mask(data["floor"] == "1, 2 этаж", 1)
        .mask(data["floor"] == "подвал, 1, 2", 1)
        .mask(data["floor"] == "1-7", 1)
        .mask(data["floor"] == "1 (по док-м цоколь)", 1)
        .mask(data["floor"] == "1,2,подвал ", 1)
        .mask(data["floor"] == "подвал, 2", 2)
        .mask(data["floor"] == "подвал,1,2,3", 1)
        .mask(data["floor"] == "1,2,3 этаж, подвал ", 1)
        .mask(data["floor"] == "1,2,3 этаж, подвал", 1)
        .mask(data["floor"] == "2, 3, 4, тех.этаж", 2)
        .mask(data["floor"] == "цокольный, 1,2", 1)
        .mask(data["floor"] == "Техническое подполье", -1)
        .mask(data["floor"] == "1.2", 1)
        .astype(float)
    )
    return data


THRESHOLD = 0.15
NEGATIVE_WEIGHT = 1.1

EPS = 1e-8

UNKNOWN_VALUE = "missing"


class PriceTypeEnum(IntEnum):
    OFFER_PRICE = 0  # цена из объявления
    MANUAL_PRICE = 1  # цена, полученная путем ручной оценки


def prepare_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет пропущенные категориальные переменные
    :param df: dataframe, обучающая выборка
    :return: dataframe
    """
    df_new = df.copy()
    fillna_cols = ["region", "city", "street", "realty_type"]
    df_new[fillna_cols] = df_new[fillna_cols].fillna(UNKNOWN_VALUE)

    return df_new


class CoeffBoostingModel:
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

    def __init__(
        self,
        numerical_features: typing.List[str],
        ohe_categorical_features: typing.List[str],
        ste_categorical_features: typing.List[typing.Union[str, typing.List[str]]],
        model_params: typing.Dict[str, typing.Union[str, int, float]],
    ):
        self.num_features = numerical_features
        self.ohe_cat_features = ohe_categorical_features
        self.ste_cat_features = ste_categorical_features

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.num_features),
                ("ohe", OneHotEncoder(), self.ohe_cat_features),
                (
                    "ste",
                    CatBoostEncoder(handle_missing="value", handle_unknown="value"),
                    # OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1), # CatBoostEncoder(handle_missing='value', handle_unknown='value'),
                    self.ste_cat_features,
                ),
            ]
        )

        self.model = LGBMRegressor(**model_params)

        self.pipeline = Pipeline(steps=[("preprocessor", self.preprocessor), ("model", self.model)])

        self._is_fitted = False

        self.coeff_model = LGBMRegressor(**model_params)

        self.coef_preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.num_features + ["predictions"]),
                ("ohe", OneHotEncoder(), self.ohe_cat_features),
                (
                    "ste",
                    CatBoostEncoder(handle_missing="value", handle_unknown="value"),
                    # OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1), # CatBoostEncoder(handle_missing='value', handle_unknown='value'),
                    self.ste_cat_features,
                ),
            ]
        )

        self.coeff_pipeline = Pipeline(steps=[("preprocessor", self.coef_preprocessor), ("model", self.coeff_model)])

    def _find_corr_coefficient(self, X_manual: pd.DataFrame, y_manual: pd.Series):
        """Вычисление корректирующего коэффициента

        :param X_manual: pd.DataFrame с ручными оценками
        :param y_manual: pd.Series - цены ручника
        """
        X_manual["predictions"] = self.pipeline.predict(X_manual)
        self.coeff_pipeline.fit(
            X_manual,
            y_manual,
            model__feature_name=[f"{i}" for i in range(X_manual.shape[1])],
            model__sample_weight=1 / y_manual.values,
        )
        self.__is_fitted = True

    def fit(self, X_offer: pd.DataFrame, y_offer: pd.Series, X_manual: pd.DataFrame, y_manual: pd.Series):
        """Обучение модели.
        ML модель обучается на данных по предложениям на рынке (цены из объявления)
        Затем вычисляется среднее отклонение между руяными оценками и предиктами для корректировки стоимости

        :param X_offer: pd.DataFrame с объявлениями
        :param y_offer: pd.Series - цена предложения (в объявлениях)
        :param X_manual: pd.DataFrame с ручными оценками
        :param y_manual: pd.Series - цены ручника
        """
        print("Fit lightgbm")
        self.pipeline.fit(
            X_offer,
            np.log1p(y_offer),
            model__feature_name=[f"{i}" for i in range(X_offer.shape[1])],
            model__sample_weight=1 / np.log1p(y_offer.values),
        )  # ,model__categorical_feature=None)
        print("Find corr coefficient")
        self._find_corr_coefficient(X_manual, np.log1p(y_manual))
        self.__is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.array:  # type: ignore
        """Предсказание модели Предсказываем преобразованный таргет, затем конвертируем в обычную цену через обратное
        преобразование.

        :param X: pd.DataFrame
        :return: np.array, предсказания (цены на коммерческую недвижимость)
        """
        if self.__is_fitted:
            X["predictions"] = self.pipeline.predict(X)
            predictions = np.expm1(np.clip(self.coeff_pipeline.predict(X), 5, 15))
            return predictions  # type: ignore
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
    def load(cls, path: str):
        """Сериализует модель в pickle.

        :param path: str, путь до файла
        :return: Модель
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model


@dataclass
class SearchConfig:
    study_name: str = "lgbm"
    n_trials: int = 100


cs = ConfigStore.instance()
cs.store(name="SearchConfig", node=SearchConfig)


N_JOBS = 8


@hydra.main(config_path=None, config_name="SearchConfig")
def main(cfg: SearchConfig):
    train_df = pd.read_csv(train_path)
    train_df["month"] = pd.to_datetime(train_df["date"]).dt.month
    train_df["month_num"] = pd.to_datetime(train_df["date"]).dt.month
    train_df = correct_floor(train_df)
    train_df_new = train_df

    def objective(trial: Trial):
        MODEL_PARAMS = dict(
            n_estimators=trial.suggest_int("n_estimators", low=2500, high=10000, step=100),
            learning_rate=trial.suggest_float("learning_rate", low=1e-4, high=1e-1, log=True),
            lambda_l1=trial.suggest_float("lambda_l1", low=1e-4, high=5, log=True),
            lambda_l2=trial.suggest_float("lambda_l2", low=1e-4, high=5, log=True),
            num_leaves=trial.suggest_int("num_leaves", low=32, high=1024, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", low=20, high=100),
            min_gain_to_split=trial.suggest_float("min_gain_to_split", low=1e-4, high=1e-1, log=True),
            feature_fraction=trial.suggest_float("feature_fraction", low=0.5, high=1.0),
            bagging_freq=trial.suggest_int("bagging_freq", low=5, high=100),
            importance_type="gain",
            n_jobs=N_JOBS,
            random_state=563,
            objective=trial.suggest_categorical("lgbm_objective", choices=["huber", "quantile", "regression_l1"]),
            bagging_fraction=trial.suggest_float("bagging_fraction", low=0.5, high=0.99),
        )

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=239)
        metrics_arr = []
        predicts_arr = []
        for fold, (tr, va) in enumerate(kf.split(train_df_new, train_df_new["price_type"])):
            df_tr = prepare_categorical(train_df_new.loc[tr].reset_index(drop=True))
            df_vl = prepare_categorical(train_df_new.loc[va].reset_index(drop=True))

            X_offer_tr = df_tr[df_tr.price_type == PriceTypeEnum.OFFER_PRICE][
                NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES
            ]
            y_offer_tr = df_tr[df_tr.price_type == PriceTypeEnum.OFFER_PRICE][TARGET]

            X_manual_tr = df_tr[df_tr.price_type == PriceTypeEnum.MANUAL_PRICE][
                NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES
            ]
            y_manual_tr = df_tr[df_tr.price_type == PriceTypeEnum.MANUAL_PRICE][TARGET]

            X_manual_vl = df_vl[df_vl.price_type == PriceTypeEnum.MANUAL_PRICE][
                NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES
            ]
            y_manual_vl = df_vl[df_vl.price_type == PriceTypeEnum.MANUAL_PRICE][TARGET]
            X_manual_vl_id = df_vl[df_vl.price_type == PriceTypeEnum.MANUAL_PRICE]["id"]

            model = CoeffBoostingModel(
                numerical_features=NUM_FEATURES,  # type: ignore
                ohe_categorical_features=CATEGORICAL_OHE_FEATURES,  # type: ignore
                ste_categorical_features=CATEGORICAL_STE_FEATURES,  # type: ignore
                model_params=MODEL_PARAMS,  # type: ignore
            )

            model.fit(X_offer_tr, y_offer_tr, X_manual_tr, y_manual_tr)

            predictions_manual = model.predict(X_manual_vl)
            metrics = metrics_stat(y_manual_vl.values, predictions_manual)
            predicts_arr.append(pd.DataFrame((X_manual_vl_id, predictions_manual), columns=["id", f"predict_{fold}"]))
            print(f"fold: {fold}, metrics {metrics}")
            metrics_arr.append(metrics)
            model.save(f"model_bst_11_{fold}.bin")

        score = np.mean([e["raif_metric"] for e in metrics_arr]).item()

        return score

    OPTUNA_USER = os.getenv("OPTUNA_USER")
    OPTUNA_PASSWORD = os.getenv("OPTUNA_PASSWORD")
    OPTUNA_HOST = os.getenv("OPTUNA_HOST")
    OPTUNA_PORT = os.getenv("OPTUNA_PORT")
    OPTUNA_DATABASE = os.getenv("OPTUNA_DATABASE")

    storage = str(
        sqlalchemy.engine.url.URL(
            drivername="mysql+pymysql",
            username=OPTUNA_USER,
            password=OPTUNA_PASSWORD,
            host=OPTUNA_HOST,
            port=OPTUNA_PORT,
            database=OPTUNA_DATABASE,
        )
    )

    study = create_study(storage=storage, study_name=cfg.study_name, load_if_exists=True, direction="minimize")

    study.optimize(objective, n_trials=cfg.n_trials)


if __name__ == "__main__":
    load_dotenv()
    main()
