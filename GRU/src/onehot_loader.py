import pandas as pd
import numpy as np

def cargar_csv_onehot(ruta_csv, columnas_target=None, return_dataframe=False):
    """
    Carga un CSV y aplica One-Hot Encoding a entradas (X) y salidas (Y) de forma automática.
    
    - X: categóricas → one-hot
    - Y: categóricas → one-hot, continuas → float32
    """
    print("Cargando CSV y aplicando One-Hot Encoding...")
    df = pd.read_csv(ruta_csv)

    # =========================
    # 🎯 TARGET (Y)
    # =========================
    target_info = {"categorical": [], "continuous": [], "onehot_cols": {}}
    Y = None

    if columnas_target is not None:
        df_target = df[columnas_target].copy()
        df_continuous = df_target.select_dtypes(exclude=["object", "category"])
        df_categorical = df_target.select_dtypes(include=["object", "category"])

        # One-hot para categóricas
        Y_parts = []
        for col in df_categorical.columns:
            print(f"[Y] {col} → categórica → One-Hot")
            dummies = pd.get_dummies(df_categorical[col], prefix=col)
            Y_parts.append(dummies)
            target_info["categorical"].append(col)
            target_info["onehot_cols"][col] = dummies.columns.tolist()

        # Continuas → float32
        for col in df_continuous.columns:
            print(f"[Y] {col} → continua (float32)")
            Y_parts.append(df_continuous[col].astype(np.float32).to_frame())
            target_info["continuous"].append(col)

        df_target_final = pd.concat(Y_parts, axis=1)
        Y = df_target_final.to_numpy(dtype=np.float32)
        df = df.drop(columns=columnas_target)

    # =========================
    # 📊 INPUTS (X)
    # =========================
    categorical_cols_X = df.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"[X] Columnas categóricas: {categorical_cols_X}")

    df_onehot_X = pd.get_dummies(df)
    categorical_info_X = {}
    for cat_col in categorical_cols_X:
        cat_onehot_cols = [c for c in df_onehot_X.columns if c.startswith(cat_col + "_")]
        categorical_info_X[cat_col] = cat_onehot_cols

    X = df_onehot_X if return_dataframe else df_onehot_X.to_numpy(dtype=np.float32)

    print("One-Hot Encoding aplicado correctamente a X y Y.")
    return X, Y, categorical_info_X, df_onehot_X.columns.tolist(), target_info