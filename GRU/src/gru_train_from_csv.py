
import argparse

parser = argparse.ArgumentParser(description="Script para entrenar GRU")

parser.add_argument("--csv", type=str, required=True, help="Ruta del archivo CSV de entrada")
parser.add_argument("--epochs", type=int, default=100, help="Número de épocas")
parser.add_argument("--onehot", type=bool, default=False, help="Indica si necesita aplicar onehot")

args = parser.parse_args()

# ============================================================
# 🔧 CONFIGURACIÓN
# ============================================================

CSV_PATH = "datatest/"+args.csv

NEED_ONEHOT = args.onehot

HIDDEN_SIZE = 64
NUM_LAYERS = 1
OUTPUT_SIZE = 6
SEQUENCE_LENGTH = 35

BATCH_SIZE = 32
EPOCHS = args.epochs
LEARNING_RATE = 0.01

NOISE_STD = 0.05
ACCURACY_THRESHOLD = 0.1

USE_CUDA = True


# ============================================================
# 📦 IMPORTS
# ============================================================

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from onehot_loader import cargar_csv_onehot
import os

# Crea el directorio si no existe
os.makedirs("models", exist_ok=True)


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
# 🧠 GRU SUPERVISADO
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
        out = self.fc(h)

        if self.training and NOISE_STD > 0:
            out = torch.clamp(out + torch.randn_like(out) * NOISE_STD, 0, 1)

        return out

# ============================================================
# 🎯 ENTRENAR GRU
# ============================================================

def train_gru(device, dataset, loader):
    input_size = dataset.inputs.shape[1]

    model = GRUEmotionModel(input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    print("▶ Entrenando GRU")

    for epoch in range(EPOCHS):
        loss_total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss {loss_total:.4f}")

    #Export en formato .pth
    torch.save(model.state_dict(), "models/gru_model.pth")

    print("✅ GRU supervisado guardado en models/gru_model.pth")

    return model


# ============================================================
# 📊 EVALUACIÓN + MATRIZ CONFUSIÓN
# ============================================================

def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            preds.append(model(x).cpu().numpy())
            targets.append(y.cpu().numpy())

    preds = np.vstack(preds)
    targets = np.vstack(targets)

    accuracy = np.mean(np.abs(preds - targets) < ACCURACY_THRESHOLD)

    correlations = [
        np.corrcoef(preds[:, i], targets[:, i])[0, 1]
        for i in range(OUTPUT_SIZE)
        if np.std(targets[:, i]) > 0
    ]

    # -------- MATRIZ CONFUSIÓN POR EMOCIÓN --------
    threshold = 0.5
    emotion_names = ["Ira", "Miedo", "Felicidad", "Tristeza", "Sorpresa", "Disgusto"]

    for i, emotion in enumerate(emotion_names):
        y_true = (targets[:, i] >= threshold).astype(int)
        y_pred = (preds[:, i] >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])  # <-- Corregido
        print(f"\n🧩 Matriz de Confusión - {emotion}")
        print(cm)

    # -------- MATRIZ CONFUSIÓN -------- 
    y_true = np.argmax(targets, axis=1)
    y_pred = np.argmax(preds, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    print("\n🧩 Matriz de Confusión")
    print(cm)
    

    print("\n📊 MÉTRICAS")
    print(f"Precisión (tolerancia {ACCURACY_THRESHOLD}): {accuracy:.4f}")
    print(f"Correlación media: {np.mean(correlations):.4f}")

    plt.figure()
    plt.imshow(cm)
    plt.title("Matriz de Confusión")
    plt.colorbar()
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

# ============================================================
# 📁 EXPORTAR A ONNX
# ============================================================
def export_to_onnx(model, dataset, device):
    model.eval()

    input_size = dataset.inputs.shape[1]

    # Dummy input (batch=1, seq=35, features)
    dummy_input = torch.randn(
        1,
        SEQUENCE_LENGTH,
        input_size,
        device=device
    )

    onnx_path = "models/gru_model.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

    print(f"✅ Modelo ONNX exportado en {onnx_path}")

# ============================================================
# 🏁 MAIN
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")

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
        
        X_raw, Y_raw, categorical_info = cargar_csv_onehot(
        ruta_csv=CSV_PATH,
        columnas_target=targets
        )
        dataset = EmotionSequenceDataset(X_raw, Y_raw, SEQUENCE_LENGTH)
    else:
        dataset = EmotionSequenceDataset.from_csv(CSV_PATH, SEQUENCE_LENGTH)

    loader = DataLoader(dataset, BATCH_SIZE, shuffle=True)

    model = train_gru(device, dataset, loader)
    
    export_to_onnx(model, dataset, device)

    evaluate(model, loader, device)
