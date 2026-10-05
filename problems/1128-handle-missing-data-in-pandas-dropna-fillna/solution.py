def solution(df):
    df = df.copy()

    # 1. drop sparse columns
    df = df.loc[:, df.isna().mean() <= 0.5]

    # 2. drop sparse rows (filter here, while NaNs still exist)
    df = df.loc[df.isna().mean(axis=1) <= 0.5]

    # 3. impute the rest
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            modes = df[col].mode()
            if not modes.empty:
                df[col] = df[col].fillna(modes.iloc[0])

    return df