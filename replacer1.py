import pandas as pd

def missing_value_treatment(df: pd.DataFrame, datetime_fill_method: str = 'ffill') -> pd.DataFrame:
    """
    Treat missing values in a DataFrame by:
    - Filling categorical columns with their mode.
    - Filling numerical columns with their median.
    - Filling datetime columns using forward fill (default) or backward fill.

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
    cat = list(df_copy.columns[df_copy.dtypes == "object"])
    num = list(df_copy.columns[df_copy.dtypes != "object"])
    datetime_cols = list(df_copy.columns[df_copy.dtypes == "datetime64[ns]"])

    for col in df_copy.columns:
        if col in cat:
            # Fill categorical columns with mode
            mode = df_copy[col].mode().dropna()
            if not mode.empty:
                df_copy[col] = df_copy[col].fillna(mode[0])
        elif col in datetime_cols:
            # Fill datetime columns with specified method
            if datetime_fill_method not in ['ffill', 'bfill']:
                raise ValueError("Invalid datetime_fill_method. Use 'ffill' or 'bfill'.")
            df_copy[col] = df_copy[col].fillna(method=datetime_fill_method)
        else:
            # Fill numerical columns with median
            median = df_copy[col].median()
            if pd.notna(median):
                df_copy[col] = df_copy[col].fillna(median)

    return df_copy