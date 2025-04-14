# Scikit Learn Approach
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def sklearn_replace_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace missing values in a DataFrame using Scikit-learn pipelines.

    This function handles missing values for numerical and categorical/datetime columns separately:
    - Numerical columns: Missing values are replaced with the median.
    - Categorical and datetime columns: Missing values are replaced with the most frequent value.

    Parameters:
    df (pd.DataFrame): Input DataFrame with potential missing values.

    Returns:
    pd.DataFrame: DataFrame with missing values replaced.
    """
    # Identify column types
    cat_datetime = list(df.select_dtypes(include=["object", "datetime"]).columns)
    con = list(df.select_dtypes(exclude=["object", "datetime"]).columns)

    # Define pipelines for each column type
    num_pipe = make_pipeline(SimpleImputer(strategy="median"))
    cat_datetime_pipe = make_pipeline(SimpleImputer(strategy="most_frequent"))

    # Combine pipelines into a ColumnTransformer
    replacer = ColumnTransformer(
        [
            ("num", num_pipe, con),
            ("cat_datetime", cat_datetime_pipe, cat_datetime),
        ]
    ).set_output(transform="pandas")

    # Apply transformations
    df_clone = df.copy()
    df_clone = replacer.fit_transform(df_clone)
    df_clone.columns = [col.split("__")[-1] for col in df_clone.columns]
    df_clone = df_clone[df.columns]
    return df_clone
