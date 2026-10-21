"""Decide what is known at checkout, then keep the test data safe."""

from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 7
NUMERIC_FEATURES = ["distance_km", "items", "hour"]
CATEGORICAL_FEATURES = ["weather", "day_type"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
EXCLUDED_FEATURES = [
    "order_id",
    "vehicle",
    "prep_minutes",
    "courier_months",
    "customer_rating",
]
TARGET = "late"


def split_orders(orders):
    """Use 75% for learning, and hide 25% for one final test."""
    features = orders[FEATURE_COLUMNS].copy()
    answers = orders[TARGET].copy()
    return train_test_split(
        features,
        answers,
        test_size=0.25,
        random_state=SEED,
        stratify=answers,
    )


def training_folds():
    """Five fair, repeatable folds only for the training rows."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def build_pipeline(model):
    """Fill numbers, scale, encode words, then let the model learn."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    number_steps = Pipeline(
        [
            ("fill_missing", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    word_steps = OneHotEncoder(handle_unknown="ignore")
    prepare = ColumnTransformer(
        [
            ("numbers", number_steps, NUMERIC_FEATURES),
            ("words", word_steps, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("prepare", prepare), ("model", model)])
