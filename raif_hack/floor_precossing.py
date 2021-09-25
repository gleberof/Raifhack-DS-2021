import numpy as np
import pandas as pd


def is_podval(x):
    if isinstance(x, float):
        return 0
    else:
        if "подв" in x.lower():
            return 1
        else:
            return 0


def is_tsokol(x):
    if isinstance(x, float):
        return 0
    else:
        if "цоко" in x.lower():
            return 1
        else:
            return 0


def is_antresol(x):
    if isinstance(x, float):
        return 0
    else:
        if "антре" in x.lower():
            return 1
        else:
            return 0


def is_mansarda(x):
    if isinstance(x, float):
        return 0
    else:
        if "мансар" in x.lower() or "мезонин" in x.lower():
            return 1
        else:
            return 0


def is_teknicheskii(x):
    if isinstance(x, float):
        return 0
    else:
        if "тех" in x.lower():
            return 1
        else:
            return 0


def is_floor(x):
    try:
        x = float(x)
        if np.isnan(x):
            return 0
        return 1
    except ValueError:
        return 0


def get_floor(x):
    try:
        x = float(x)
        if np.isnan(x):
            return 0
        return x
    except ValueError:
        return 0


def is_first_floor(x):
    try:
        x = float(x)
        if x == 1:
            return 1
        else:
            return 0
    except ValueError:
        return 0


def is_second_floor(x):
    try:
        x = float(x)
        if x == 2:
            return 1
        else:
            return 0
    except ValueError:
        return 0


def is_third_floor(x):
    try:
        x = float(x)
        if x == 3:
            return 1
        else:
            return 0
    except ValueError:
        return 0


def floor_processing(floor_feature):
    floor_features = []

    floor_features.append(floor_feature.isna().astype(int))
    floor_features.append(floor_feature.apply(is_podval))
    floor_features.append(floor_feature.apply(is_tsokol))
    floor_features.append(floor_feature.apply(is_antresol))
    floor_features.append(floor_feature.apply(is_mansarda))
    floor_features.append(floor_feature.apply(is_teknicheskii))
    floor_features.append(floor_feature.apply(is_floor))
    floor_features.append(floor_feature.apply(get_floor).astype(int))
    floor_features.append(floor_feature.apply(is_first_floor))
    floor_features.append(floor_feature.apply(is_second_floor))
    floor_features.append(floor_feature.apply(is_third_floor))
    floor_features = pd.concat(floor_features, axis=1)
    floor_features.columns = [
        "floor_is_nan",
        "floor_is_podval",
        "floor_is_tsokol",
        "floor_is_antresol",
        "floor_is_mansarda",
        "floor_is_teknicheskii",
        "floor_is_floor",
        "floor_number",
        "floor_is_first_floor",
        "floor_is_second_floor",
        "floor_is_third_floor",
    ]
    return floor_features
