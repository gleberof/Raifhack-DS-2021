import logging

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer  # noqa
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from raif_hack.settings import CATEGORICAL_OHE_FEATURES, NUM_FEATURES

logger = logging.getLogger(__name__)

# all given numeric features distributions should be mapped to log space

numeric_transformer = Pipeline(
    steps=[
        ("log_space", FunctionTransformer(func=np.log1p, inverse_func=np.expm1)),
        ("imputer", SimpleImputer(strategy="median")),
        # ("imputer", KNNImputer()),
        ("scaler", StandardScaler()),
    ]
)


# all given categorical features have low cardinality and can be one-hot-encoded
categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)


def get_preprocessor(numeric_features=NUM_FEATURES, categorical_ohe_features=CATEGORICAL_OHE_FEATURES):
    preprocessor = Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        ("numeric", numeric_transformer, numeric_features),
                        ("categorical", categorical_transformer, categorical_ohe_features),
                    ]
                ),
            )
        ]
    )

    return preprocessor
