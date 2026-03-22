import pandas as pd
import numpy as np


def cargar_csv_onehot(
    ruta_csv,
    columnas_target=None,
    return_dataframe=False
):
    """
    Carga un CSV y aplica One-Hot Encoding.
    NO crea tensores.

    Returns:
        X (np.ndarray | pd.DataFrame)
        Y (np.ndarray | None)
    """
    print("Aplicando One-Hot Encoding...")
    df = pd.read_csv(ruta_csv)

    if columnas_target is not None:
        Y = df[columnas_target].to_numpy(dtype=np.float32)
        df = df.drop(columns=columnas_target)
    else:
        Y = None

    # Detectar columnas categóricas automáticamente
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"Columnas categoricas: {categorical_cols}")
    df_onehot = pd.get_dummies(df)

    # Guardar info de qué columnas corresponden a cada variable categórica
    categorical_info = {}
    for cat_col in categorical_cols:
        cat_onehot_cols = [c for c in df_onehot.columns if c.startswith(cat_col + "_")]
        categorical_info[cat_col] = cat_onehot_cols

    if return_dataframe:
        X = df_onehot
    else:
        X = df_onehot.to_numpy(dtype=np.float32)

    print("One-Hot Encoding aplicado.")
    return X, Y, categorical_info, df_onehot.columns.tolist()