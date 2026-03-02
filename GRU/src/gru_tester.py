import argparse

parser = argparse.ArgumentParser(description="Script para entrenar GRU")

parser.add_argument("--csv", type=str, required=True, help="Ruta del archivo CSV de entrada")
parser.add_argument("--onehot", type=bool, default=False, help="Indica si necesita aplicar onehot")

args = parser.parse_args()

# ============================================================
# 🔧 CONFIGURACIÓN
# ============================================================

CSV_PATH = "datatest/"+args.csv
MODEL_PATH = "models/gru_model.pth"
CSV_OUTPUT = "datatest/predicted.csv"

OUTPUT_SIZE = 6
SEQUENCE_LENGTH = 35

HIDDEN_SIZE = 64
NUM_LAYERS = 1

BATCH_SIZE = 32
ACCURACY_THRESHOLD = 0.1
USE_CUDA = True
NEED_ONEHOT = args.onehot

# ============================================================
# 📦 IMPORTS
# ============================================================

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from onehot_loader import cargar_csv_onehot

# ============================================================
# 📊 DATASET
# ============================================================

class EmotionSequenceDataset(Dataset):
    def __init__(self, X_raw, Y_raw, sequence_length):
        self.inputs = X_raw
        self.targets = Y_raw
        self.sequence_length = sequence_length
        
    @classmethod
    def from_csv(cls, csv_path, sequence_length):
        df = pd.read_csv(csv_path)
        inputs = df.iloc[:, :-OUTPUT_SIZE].values
        targets = df.iloc[:, -OUTPUT_SIZE:].values
        return cls(inputs, targets, sequence_length)

    def __len__(self):
        return len(self.inputs) - self.sequence_length

    def __getitem__(self, idx):
        x = self.inputs[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length - 1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ============================================================
# 🧠 MODELO GRU (MISMA ESTRUCTURA)
# ============================================================

class GRUEmotionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.gru = nn.GRU(input_size, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, h = self.gru(x)
        h = h[-1]
        return self.fc(h)

# ============================================================
# 📊 EVALUACIÓN
# ============================================================

def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds.append(model(x).cpu().numpy())
            targets.append(y.numpy())

    preds = np.vstack(preds)
    targets = np.vstack(targets)

    # ---------- PRECISIÓN ----------
    accuracy = np.mean(np.abs(preds - targets) < ACCURACY_THRESHOLD)
    print(f"\n✅ Precisión (tolerancia ±{ACCURACY_THRESHOLD}): {accuracy:.4f}")

    # ---------- CORRELACIÓN ----------
    emotion_names = ["Ira", "Miedo", "Felicidad", "Tristeza", "Sorpresa", "Disgusto"]
    correlations = []

    print("\n📈 Correlación por emoción:")
    for i, name in enumerate(emotion_names):
        if np.std(targets[:, i]) == 0:
            print(f"  ⚠️ {name}: constante (correlación no definida)")
            correlations.append(np.nan)
        else:
            corr, _ = pearsonr(targets[:, i], preds[:, i])
            correlations.append(corr)
            print(f"  {name}: {corr:.4f}")

    valid_corrs = [c for c in correlations if not np.isnan(c)]
    if valid_corrs:
        print(f"\n📊 Correlación media: {np.mean(valid_corrs):.4f}")
    else:
        print("\n📊 Correlación media: no definida")


def save_predictions_csv(model, loader, device, csv_output="predictions.csv"):
    model.eval()
    all_inputs, all_preds = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            y_hat = model(x).cpu().numpy()
            all_preds.append(y_hat)
            all_inputs.append(x.cpu().numpy())

    # Concatenar todos los batches
    all_inputs = np.concatenate(all_inputs, axis=0)  # (N, SEQ_LEN, input_dim)
    all_preds = np.vstack(all_preds)                 # (N, OUTPUT_SIZE)

    # Tomar la última fila de cada secuencia
    all_inputs_last = all_inputs[:, -1, :]           # (N, input_dim)

    # Crear DataFrame
    df = pd.DataFrame(
        np.concatenate([all_inputs_last, all_preds], axis=1),
        columns=[f"Input_{i+1}" for i in range(all_inputs_last.shape[1])] +
                ["Ira","Miedo","Felicidad","Tristeza","Sorpresa","Disgusto"]
    )

    df.to_csv(CSV_OUTPUT, index=False)
    print(f"✅ Predicciones guardadas en {CSV_OUTPUT}")


# ============================================================
# 🏁 MAIN
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    print(f"🖥️ Usando dispositivo: {device}")

    # Dataset
    dataset = None
    if(NEED_ONEHOT):
        targets = [
        "Ira",
        "Miedo",
        "Felicidad",
        "Tristeza",
        "Sorpresa",
        "Disgusto"
        ]
        print(f"{CSV_PATH}")
        X_raw, Y_raw, categorical_info, feature_columns = cargar_csv_onehot(
        ruta_csv=CSV_PATH,
        columnas_target=targets
        )
        dataset = EmotionSequenceDataset(X_raw, Y_raw, SEQUENCE_LENGTH)
    else:
        dataset = EmotionSequenceDataset.from_csv(CSV_PATH, SEQUENCE_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Modelo
    input_size = dataset.inputs.shape[1]
    model = GRUEmotionModel(input_size).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("✅ Modelo GRU cargado correctamente")

    # Evaluación
    evaluate(model, loader, device)
    save_predictions_csv(model, loader, device)
