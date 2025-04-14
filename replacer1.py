import pandas as pd


def missing_value_treatment(
    df: pd.DataFrame, datetime_fill_method: str = "ffill"
) -> pd.DataFrame:
    """
    Treat missing values in a DataFrame by:
    - Filling categorical and datetime columns with their mode.
    - Filling numerical columns with their median.

    Args:
        df (pd.DataFrame): Input DataFrame.
        datetime_fill_method (str): Method to fill datetime columns ('ffill' or 'bfill').

    Returns:
        pd.DataFrame: A new DataFrame with missing values treated.
    """
    if df.empty:
        raise ValueError("The input DataFrame is empty.")

    # Clone the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Identify column types
    cat_datetime = list(df_copy.select_dtypes(include=["object", "datetime"]).columns)
    con = list(df_copy.select_dtypes(exclude=["object", "datetime"]).columns)

    for col in df_copy.columns:
        if col in cat_datetime:
            # Fill categorical and datetime columns with mode
            mode = df_copy[col].mode().dropna()
            if not mode.empty:
                df_copy[col] = df_copy[col].fillna(mode[0])
        else:
            # Fill numerical columns with median
            median = df_copy[col].median()
            if pd.notna(median):
                df_copy[col] = df_copy[col].fillna(median)

    return df_copy
