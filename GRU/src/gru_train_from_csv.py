# ============================================================
# gru_emocion_intensidad.py
# ============================================================

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from onehot_loader import cargar_csv_onehot

# ============================================================
# 🔧 ARGUMENTOS
# ============================================================

parser = argparse.ArgumentParser(description="GRU supervisado para Emoción + Intensidad")
parser.add_argument("--csv", type=str, required=True, help="Archivo CSV de entrada")
parser.add_argument("--epochs", type=int, default=100, help="Número de épocas")
parser.add_argument("--onehot", type=bool, default=False, help="Aplicar one-hot a la emoción")
args = parser.parse_args()

# ============================================================
# 🔧 CONFIGURACIÓN
# ============================================================

CSV_PATH = os.path.join("datatest", args.csv)
USE_CUDA = True
DEVICE = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")

HIDDEN_SIZE = 64
NUM_LAYERS = 1
SEQUENCE_LENGTH = 35
BATCH_SIZE = 32
LEARNING_RATE = 0.001
ACCURACY_THRESHOLD = 0.1

EMOTIONS = ["Ira","Miedo","Felicidad","Tristeza","Sorpresa","Disgusto", "Neutra"]
OUTPUT_SIZE = len(EMOTIONS) + 1  # emoción one-hot + intensidad

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
# 🧠 GRU supervisado (Emoción + Intensidad)
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
        intensity = self.fc_intensity(h_last)
        return logits_emotion, intensity

# ============================================================
# 🎯 ENTRENAMIENTO
# ============================================================

def train_gru(device, dataset, loader, n_emotions):
    input_size = dataset.inputs.shape[1]
    model = GRUEmotionIntensity(input_size, n_emotions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn_emotion = nn.CrossEntropyLoss()
    loss_fn_intensity = nn.MSELoss()

    print("▶ Entrenando GRU supervisado (Emoción + Intensidad)")

    for epoch in range(args.epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits_emotion, intensity_pred = model(x)
            emotion_true = torch.argmax(y[:, :n_emotions], dim=-1)
            intensity_true = y[:, n_emotions:]

            loss = loss_fn_emotion(logits_emotion, emotion_true) + loss_fn_intensity(intensity_pred, intensity_true)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch+1) % 10 == 0:
            print(f"GRU Epoch {epoch+1}/{args.epochs} - Loss {total_loss:.4f}")

    torch.save(model.state_dict(), "models/gru_model.pth")
    print("✅ Modelo guardado en models/gru_model.pth")
    return model

# ============================================================
# 📊 EVALUACIÓN
# ============================================================

def evaluate(model, loader, device, n_emotions):
    model.eval()
    preds_list, targets_list = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits_emotion, intensity_pred = model(x)

            # Convertir logits a one-hot
            emotion_onehot = torch.zeros_like(logits_emotion).scatter_(1, torch.argmax(logits_emotion, dim=1, keepdim=True), 1.0)

            preds = torch.cat([emotion_onehot, intensity_pred], dim=1)
            preds_list.append(preds.cpu().numpy())
            targets_list.append(y.cpu().numpy())

    preds = np.vstack(preds_list)
    targets = np.vstack(targets_list)

    # Precisión intensidad
    accuracy_intensity = np.mean(np.abs(preds[:, n_emotions:] - targets[:, n_emotions:]) < ACCURACY_THRESHOLD)

    # Correlación intensidad
    correlations = [np.corrcoef(preds[:, n_emotions+i].flatten(), targets[:, n_emotions+i].flatten())[0,1]
                    for i in range(1) if np.std(targets[:, n_emotions+i]) > 0]

    # Matriz de confusión emoción
    y_true = np.argmax(targets[:, :n_emotions], axis=1)
    y_pred = np.argmax(preds[:, :n_emotions], axis=1)
    cm = confusion_matrix(y_true, y_pred)

    print("\n🧩 Matriz de Confusión - Emoción")
    print(cm)
    print(f"\n📊 Precisión intensidad (tolerancia {ACCURACY_THRESHOLD}): {accuracy_intensity:.4f}")
    print(f"Correlación media intensidad: {np.mean(correlations):.4f}")

    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Matriz de Confusión - Emoción")
    plt.colorbar()
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

# ============================================================
# 📁 EXPORT A ONNX
# ============================================================

def export_to_onnx(model, dataset, device):
    model.eval()
    input_size = dataset.inputs.shape[1]
    dummy_input = torch.randn(1, SEQUENCE_LENGTH, input_size, device=device)
    onnx_path = "models/gru_model.onnx"
    torch.onnx.export(model, dummy_input, onnx_path,
                      export_params=True,
                      opset_version=18,
                      do_constant_folding=True,
                      input_names=["input"],
                      output_names=["logits_emotion","intensity"],
                      dynamic_axes={"input": {0: "batch_size"},
                                    "logits_emotion": {0: "batch_size"},
                                    "intensity": {0: "batch_size"}})
    print(f"✅ Modelo ONNX exportado en {onnx_path}")

# ============================================================
# 🏁 MAIN
# ============================================================

if __name__ == "__main__":
    # Cargar dataset
    if args.onehot:
        X_raw, Y_raw, categorical_info, feature_columns = cargar_csv_onehot(
            ruta_csv=CSV_PATH,
            columnas_target=["Emocion","Intensidad"]
        )
        dataset = EmotionSequenceDataset(X_raw, Y_raw, SEQUENCE_LENGTH)
    else:
        dataset = EmotionSequenceDataset.from_csv(CSV_PATH, SEQUENCE_LENGTH)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Entrenar GRU
    model = train_gru(DEVICE, dataset, loader, n_emotions=len(EMOTIONS))

    # Exportar a ONNX
    export_to_onnx(model, dataset, DEVICE)

    # Evaluar
    evaluate(model, loader, DEVICE, n_emotions=len(EMOTIONS))