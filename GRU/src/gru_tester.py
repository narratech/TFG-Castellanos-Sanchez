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

SEQUENCE_LENGTH = 35

HIDDEN_SIZE = 64
NUM_LAYERS = 1

BATCH_SIZE = 32
ACCURACY_THRESHOLD = 0.1
USE_CUDA = True
NEED_ONEHOT = args.onehot

EMOTIONS = ["Ira","Miedo","Felicidad","Tristeza","Sorpresa","Disgusto", "Neutra"]
OUTPUT_SIZE = len(EMOTIONS) + 1  # emoción one-hot + intensidad

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

class GRUEmotionIntensity(nn.Module):
    def __init__(self, input_size, n_emotions):
        super().__init__()
        self.gru = nn.GRU(input_size, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.fc_emotion = nn.Linear(HIDDEN_SIZE, n_emotions)  # logits
        self.fc_intensity = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        _, h = self.gru(x)
        h_last = h[-1]
        logits_emotion = self.fc_emotion(h_last)  # ⚠ sin softmax
        intensity = torch.sigmoid(self.fc_intensity(h_last))
        return logits_emotion, intensity

# ============================================================
# 📊 EVALUACIÓN
# ============================================================

def evaluate(model, loader, device):
    model.eval()
    preds_list, targets_list = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float()

            # 👇 desempaquetar
            logits_emotion, intensity_pred = model(x)

            # ---------- emociones → one-hot ----------
            emotion_preds = logits_emotion.cpu().numpy()
            emotion_preds_discrete = np.zeros_like(emotion_preds)
            emotion_preds_discrete[
                np.arange(emotion_preds.shape[0]),
                np.argmax(emotion_preds, axis=1)
            ] = 1

            # ---------- intensidad ----------
            intensity_pred = intensity_pred.cpu().numpy()

            # ---------- concatenar ----------
            preds_batch = np.concatenate([emotion_preds_discrete, intensity_pred], axis=1)

            preds_list.append(preds_batch)
            targets_list.append(y.cpu().numpy())

    preds = np.vstack(preds_list)
    targets = np.vstack(targets_list)

    # ---------- PRECISIÓN ----------
    accuracy = np.mean(np.abs(preds - targets) < ACCURACY_THRESHOLD)
    print(f"\n✅ Precisión (tolerancia ±{ACCURACY_THRESHOLD}): {accuracy:.4f}")

    # ---------- CORRELACIÓN ----------
    emotion_names = ["Ira", "Miedo", "Felicidad", "Tristeza", "Sorpresa", "Disgusto", "Neutra", "Intensidad"]
    correlations = []

    print("\n📈 Correlación por variable:")
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


def save_predictions_csv(model, loader, device):
    model.eval()
    all_inputs, all_preds = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()

            # 👇 CORRECTO: desempaquetar
            logits_emotion, intensity_pred = model(x)

            # ---------- emociones → one-hot ----------
            emotion_preds = logits_emotion.cpu().numpy()
            emotion_preds_discrete = np.zeros_like(emotion_preds)
            emotion_preds_discrete[
                np.arange(emotion_preds.shape[0]),
                np.argmax(emotion_preds, axis=1)
            ] = 1

            # ---------- intensidad ----------
            intensity_pred = intensity_pred.cpu().numpy()

            # ---------- concatenar ----------
            preds_batch = np.concatenate([emotion_preds_discrete, intensity_pred], axis=1)

            all_preds.append(preds_batch)
            all_inputs.append(x.cpu().numpy())

    # Concatenar todos los batches
    all_inputs = np.concatenate(all_inputs, axis=0)  # (N, SEQ_LEN, input_dim)
    all_preds = np.vstack(all_preds)                 # (N, OUTPUT_SIZE)

    # Tomar la última fila de cada secuencia
    all_inputs_last = all_inputs[:, -1, :]           # (N, input_dim)

    # Columnas dinámicas correctas
    columns = (
        [f"Input_{i+1}" for i in range(all_inputs_last.shape[1])] +
        EMOTIONS +
        ["Intensidad"]
    )

    # Crear DataFrame
    df = pd.DataFrame(
        np.concatenate([all_inputs_last, all_preds], axis=1),
        columns=columns
    )

    df.to_csv(CSV_OUTPUT, index=False)
    print(f"✅ Predicciones guardadas en {CSV_OUTPUT}")

    

def evaluate_discrete(model, loader, device):
    model.eval()
    preds_list, targets_list = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float()

            # 👇 CORRECTO: desempaquetar
            logits_emotion, intensity_pred = model(x)

            # ---------- emociones → one-hot ----------
            emotion_preds = logits_emotion.cpu().numpy()
            emotion_preds_discrete = np.zeros_like(emotion_preds)
            emotion_preds_discrete[
                np.arange(emotion_preds.shape[0]),
                np.argmax(emotion_preds, axis=1)
            ] = 1

            # ---------- intensidad ----------
            intensity_pred = intensity_pred.cpu().numpy()

            # ---------- concatenar ----------
            preds_batch = np.concatenate([emotion_preds_discrete, intensity_pred], axis=1)

            preds_list.append(preds_batch)
            targets_list.append(y.cpu().numpy())

    preds = np.vstack(preds_list)
    targets = np.vstack(targets_list)

    return preds, targets


# ============================================================
# 🏁 MAIN
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    print(f"🖥️ Usando dispositivo: {device}")

    # ================= DATASET =================
    if NEED_ONEHOT:
        print(f"📄 Cargando CSV con one-hot: {CSV_PATH}")
        targets = ["Emocion","Intensidad"]
        X_raw, Y_raw, categorical_info_X, feature_columns, target_info = cargar_csv_onehot(
            ruta_csv=CSV_PATH,
            columnas_target=targets
        )
        dataset = EmotionSequenceDataset(X_raw, Y_raw, SEQUENCE_LENGTH)
    else:
        print(f"📄 Cargando CSV sin one-hot: {CSV_PATH}")
        dataset = EmotionSequenceDataset.from_csv(CSV_PATH, SEQUENCE_LENGTH)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ================= MODELO =================
    input_size = dataset.inputs.shape[1]
    model = GRUEmotionIntensity(input_size, len(EMOTIONS)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("✅ Modelo GRU cargado correctamente")

    evaluate(model, loader, device)

    save_predictions_csv(model, loader, device)
