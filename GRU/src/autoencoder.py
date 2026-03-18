# ============================================================
# autoencoder_gru_emocion_intensidad_discreto.py
# ============================================================

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from onehot_loader import cargar_csv_onehot
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ============================================================
# 🔧 ARGUMENTOS
# ============================================================

parser = argparse.ArgumentParser(description="GRU Autoencoder + GRU supervisado para Emocion e Intensidad")
parser.add_argument("--csv", type=str, required=True, help="Ruta del CSV de entrada")
parser.add_argument("--epochs", type=int, default=100, help="Número de secuencias sintéticas a generar")
args = parser.parse_args()

# ============================================================
# 🔧 CONFIGURACIÓN
# ============================================================

DATASET_PATH = args.csv
CSV_FOLDER = "datatest/"
OUTPUT_CSV = "generated_" + os.path.basename(DATASET_PATH)

SEQUENCE_LENGTH = 35
BLOCK_SIZE = 2 * SEQUENCE_LENGTH
LATENT_SIZE = 32
HIDDEN_SIZE = 64
AE_EPOCHS = 160
GRU_EPOCHS = 100
BATCH_SIZE = 32
LR = 0.01
LATENT_NOISE_STD = 0.1
USE_CUDA = True

# Emociones
EMOTIONS = ["Ira","Miedo","Felicidad","Tristeza","Sorpresa","Disgusto", "Neutra"]

# ============================================================
# 📊 DATASET
# ============================================================

class RealDataset(Dataset):
    def __init__(self, X_raw, Y_raw):
        self.X = X_raw
        self.Y = Y_raw

    def create_windows(self):
        Xw, Yw = [], []
        N = len(self.X)

        for start in range(0, N, BLOCK_SIZE):
            end = start + BLOCK_SIZE

            # evitar bloques incompletos
            if end > N:
                break

            # sliding window dentro del bloque
            for i in range(start, end - SEQUENCE_LENGTH + 1):
                x_window = self.X[i:i+SEQUENCE_LENGTH]
                y_target = self.Y[i+SEQUENCE_LENGTH-1]

                Xw.append(x_window)
                Yw.append(y_target)

        return np.array(Xw), np.array(Yw)

# ============================================================
# 🔁 AUTOENCODER GRU
# ============================================================

class GRUAutoencoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.encoder = nn.GRU(input_size, LATENT_SIZE, batch_first=True)
        self.decoder = nn.GRU(LATENT_SIZE, input_size, batch_first=True)

    def forward(self, x):
        _, h = self.encoder(x)
        z = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        x_hat, _ = self.decoder(z)
        return torch.clamp(x_hat, 0, 1)

def train_autoencoder(X, input_size, device):
    ae = GRUAutoencoder(input_size).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    loader = DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(AE_EPOCHS):
        total = 0
        for x in loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = loss_fn(ae(x), x)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"AE Epoch {epoch+1}/{AE_EPOCHS} - Loss {total:.4f}")
    return ae

def generate_sequences(ae, X_real, device, n_synth=100):
    ae.eval()
    sequences = []
    for _ in range(n_synth):
        idx = np.random.randint(len(X_real))
        x = torch.tensor(X_real[idx], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _, h = ae.encoder(x)
            z = h[-1] + torch.randn_like(h[-1]) * LATENT_NOISE_STD
            z_seq = z.unsqueeze(1).repeat(1, SEQUENCE_LENGTH, 1)
            x_new, _ = ae.decoder(z_seq)
        sequences.append(torch.clamp(x_new, 0, 1).cpu().numpy()[0])
    return np.array(sequences)

# ============================================================
# 🔹 GRU supervisado para Emocion + Intensidad
# ============================================================

class GRUEmotionIntensity(nn.Module):
    def __init__(self, input_size, n_emotions):
        super().__init__()
        self.gru = nn.GRU(input_size, HIDDEN_SIZE, batch_first=True)
        self.fc_emotion = nn.Linear(HIDDEN_SIZE, n_emotions)
        self.fc_intensity = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        _, h = self.gru(x)
        logits_emotion = self.fc_emotion(h[-1])  # ⚠ sin softmax
        intensity = self.fc_intensity(h[-1])
        return logits_emotion, intensity

def train_gru(X, Y, input_size, n_emotions, device):
    model = GRUEmotionIntensity(input_size, n_emotions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loader = DataLoader(list(zip(X,Y)), batch_size=BATCH_SIZE, shuffle=True)
    loss_fn_emotion = nn.CrossEntropyLoss()
    loss_fn_intensity = nn.MSELoss()

    for epoch in range(GRU_EPOCHS):
        total_loss = 0
        for xb, yb in loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()

            optimizer.zero_grad()

            emotion_pred, intensity_pred = model(xb)

            emotion_true = torch.argmax(yb[:, :n_emotions], dim=-1)
            intensity_true = yb[:, n_emotions:]

            loss = loss_fn_emotion(emotion_pred, emotion_true) + 0.5 * loss_fn_intensity(intensity_pred, intensity_true)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"GRU Epoch {epoch+1}/{GRU_EPOCHS} - Loss {total_loss:.4f}")
    return model

# ============================================================
# 🔹 DISCRETIZAR CATEGORICAS (0/1)
# ============================================================

def discretize_categoricals(X, categorical_info, feature_columns):
    """
    Convierte las columnas categóricas en 0 o 1 (one-hot estricto)
    X: np.array (n_sequences, seq_len, n_features)
    categorical_info: dict variable -> lista de nombres de columnas
    feature_columns: lista con todos los nombres de columnas en X
    """
    X_out = X.copy()
    col_to_idx = {c:i for i,c in enumerate(feature_columns)}

    for var, cols in categorical_info.items():
        idxs = [col_to_idx[c] for c in cols]  # <--- convertir nombres a índices
        block = X_out[:, :, idxs]
        winners = np.argmax(block, axis=-1)
        X_out[:, :, idxs] = 0
        for n in range(X.shape[0]):
            for t in range(X.shape[1]):
                X_out[n, t, idxs[winners[n,t]]] = 1
    return X_out

# ============================================================
# 💾 EXPORTAR CSV
# ============================================================

def export_csv(X_seq, preds, feature_columns, emotions):
    rows = []
    for x, p in zip(X_seq, preds):
        # reconstruir vector
        rows.append(np.concatenate([x[-1], p]))
    df = pd.DataFrame(rows, columns=feature_columns + emotions + ["Intensidad"])
    df.to_csv(CSV_FOLDER + OUTPUT_CSV, index=False)
    print(f"✅ CSV generado: {CSV_FOLDER + OUTPUT_CSV}")

# ============================================================
# 📊 PLOT ESPACIO LATENTE
# ============================================================

def plot_latents(ae, sequences, device):
    ae.eval()
    latents = []
    with torch.no_grad():
        for seq in sequences:
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            _, h = ae.encoder(x)
            latents.append(h[-1].squeeze(0).cpu().numpy())

    latents = np.array(latents)
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(latents)

    plt.figure(figsize=(8,6))
    plt.scatter(z_2d[:,0], z_2d[:,1], c=np.arange(len(z_2d)), cmap="viridis", alpha=0.7)
    plt.colorbar(label="Índice de secuencia")
    plt.title("Espacio latente – Secuencias generadas")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.grid(True)
    plt.show()

# ============================================================
# 🏁 MAIN
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")

    # Cargar CSV con one-hot aplicado a Emocion
    X_raw, Y_raw, categorical_info_X, feature_columns, target_info = cargar_csv_onehot(
        ruta_csv=os.path.join(CSV_FOLDER, DATASET_PATH),
        columnas_target=["Emocion","Intensidad"]
    )

    input_size = X_raw.shape[1]
    n_emotions = len(EMOTIONS)

    dataset = RealDataset(X_raw, Y_raw)
    X_seq, Y_seq = dataset.create_windows()

    # 1️⃣ Autoencoder
    ae = train_autoencoder(X_seq, input_size, device)
    X_synth = generate_sequences(ae, X_seq, device, n_synth=args.epochs)

    # 2️⃣ Discretizar categóricas del autoencoder
    X_synth_discrete = discretize_categoricals(X_synth, categorical_info_X, feature_columns)

    # 3️⃣ GRU supervisado
    gru = train_gru(X_seq, Y_seq, input_size, n_emotions, device)

    # 4️⃣ Predicciones GRU sobre secuencias sintéticas
    with torch.no_grad():
        logits_emotion, intensity = gru(torch.tensor(X_synth_discrete, dtype=torch.float32).to(device))
    
    # 1️⃣ Convertir emociones a one-hot
    emotion_onehot = torch.zeros_like(logits_emotion).scatter_(1, torch.argmax(logits_emotion, dim=1, keepdim=True), 1.0)

    # 2️⃣ Concatenar con intensidad
    preds = torch.cat([emotion_onehot, intensity], dim=1).cpu().numpy()

    # 5️⃣ Discretizar salida de emoción (one-hot)
    preds[:, :n_emotions] = np.eye(n_emotions)[np.argmax(preds[:, :n_emotions], axis=1)]

    # 6️⃣ Exportar CSV final
    export_csv(X_synth_discrete, preds, feature_columns, EMOTIONS)

    # 7️⃣ Plot espacio latente autoencoder
    plot_latents(ae, X_synth_discrete, device)