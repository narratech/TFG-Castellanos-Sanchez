import argparse
import configparser
import os

# ============================================================
# 📦 IMPORTS
# ============================================================

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from onehot_loader import cargar_csv_onehot

# ============================================================
# 🔧 CONFIGURACIÓN
# ============================================================

config = configparser.ConfigParser()
config.read('config.ini')

DATASET_PATH = config['Dataset']['CSV_NAME']
CSV_FOLDER = "datatest/"
OUTPUT_CSV = "generated_" + os.path.basename(DATASET_PATH)

N_SYNTHETIC = int(config['Autoencoder']['N_SYNTHETIC'])
SEQUENCE_LENGTH = int(config['Dataset']['SEQUENCE_LENGTH'])
BLOCK_SIZE = int(config['Dataset']['BLOCK_SIZE'])
LATENT_SIZE = int(config['Autoencoder']['LATENT_SIZE'])
HIDDEN_SIZE = int(config['Autoencoder']['HIDDEN_SIZE'])
NUM_LAYERS = int(config['Autoencoder']['NUM_LAYERS'])
AE_EPOCHS = int(config['Autoencoder']['AE_EPOCHS'])
GRU_EPOCHS = int(config['Autoencoder']['GRU_EPOCHS'])
BATCH_SIZE = int(config['Autoencoder']['BATCH_SIZE'])
AE_LR = float(config['Autoencoder']['AE_LEARNING_RATE'])
GRU_LR = float(config['Autoencoder']['GRU_LEARNING_RATE'])
LATENT_NOISE_STD = float(config['Autoencoder']['LATENT_NOISE_STD'])
ACCURACY_THRESHOLD = float(config['GRUTrain']['ACCURACY_THRESHOLD'])
USE_CUDA = bool(config['Autoencoder']['USE_CUDA'])

OUTPUT_COLUMNS = list(map(str, config['Dataset']['OUTPUT_NAMES'].split(',')))
OUTPUT_SIZE = len(OUTPUT_COLUMNS)


# ============================================================
# 📊 DATASET
# ============================================================

class RealDataset(Dataset):
    def __init__(self, X_raw, Y_raw):
        self.X = X_raw
        self.Y = Y_raw

    def create_windows(self):
        Xw, Yw = [], []
        for i in range(len(self.X) - SEQUENCE_LENGTH):
            Xw.append(self.X[i:i+SEQUENCE_LENGTH])
            Yw.append(self.Y[i+SEQUENCE_LENGTH-1])
        return np.array(Xw), np.array(Yw)

# ============================================================
# 🔁 AUTOENCODER GRU (SOLO INPUTS)
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
    

# ============================================================
# 🧠 GRU SUPERVISADO (SOLO REAL)
# ============================================================

class GRUEmotion(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.gru = nn.GRU(input_size, HIDDEN_SIZE, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

# ============================================================
# 🚀 ENTRENAR AUTOENCODER
# ============================================================

def train_autoencoder(X, input_size, device):
    ae = GRUAutoencoder(input_size).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=AE_LR)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        torch.tensor(X, dtype=torch.float32),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print("▶ Entrenando Autoencoder")

    for epoch in range(AE_EPOCHS):
        total = 0
        for x in loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = loss_fn(ae(x), x)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"AE Epoch {epoch+1}/{AE_EPOCHS} - Loss {total:.4f}")

    return ae

# ============================================================
# 🎲 GENERAR SECUENCIAS NUEVAS
# ============================================================

def generate_sequences(ae, X_real, device):
    ae.eval()
    sequences = []

    for _ in range(N_SYNTHETIC):
        idx = np.random.randint(len(X_real))
        x = torch.tensor(X_real[idx], dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            _, h = ae.encoder(x)
            z = h[-1] + torch.randn_like(h[-1]) * LATENT_NOISE_STD
            z_seq = z.unsqueeze(1).repeat(1, SEQUENCE_LENGTH, 1)
            x_new, _ = ae.decoder(z_seq)

        sequences.append(torch.clamp(x_new, 0, 1).cpu().numpy()[0])

    return np.array(sequences)

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

    print("\n📊 MÉTRICAS")
    print(f"Precisión (tolerancia {ACCURACY_THRESHOLD}): {accuracy:.4f}")
    print(f"Correlación media: {np.mean(correlations):.4f}")

def get_categorical_indices(categorical_info, feature_columns):
    """
    Devuelve índices reales de columnas categóricas en X
    """
    col_to_idx = {c: i for i, c in enumerate(feature_columns)}
    categorical_indices = {}

    for var, cols in categorical_info.items():
        categorical_indices[var] = [col_to_idx[c] for c in cols]

    return categorical_indices

def discretize_categoricals(X, categorical_indices):
    """
    Numéricas → continuo [0,1]
    Categóricas → one-hot estricto {0,1}
    """
    X_out = X.copy()

    for _, idxs in categorical_indices.items():
        block = X_out[:, :, idxs]          # (N, T, K)
        winners = np.argmax(block, axis=2) # (N, T)

        X_out[:, :, idxs] = 0

        for i in range(X.shape[0]):
            for t in range(X.shape[1]):
                X_out[i, t, idxs[winners[i, t]]] = 1

    return X_out

# ============================================================
# 🚀 ENTRENAR GRU (SOLO DATOS REALES)
# ============================================================

def train_gru_real(X, Y, input_size, device):
    model = GRUEmotion(input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=GRU_LR)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        list(zip(X, Y)),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print("▶ Entrenando GRU con datos reales")

    for epoch in range(GRU_EPOCHS):
        total = 0
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"GRU Epoch {epoch+1}/{GRU_EPOCHS} - Loss {total:.4f}")

    evaluate(model, loader, device)

    return model


def plot_latents(ae, sequences, device):
    ae.eval()
    latents = []

    with torch.no_grad():
        for seq in sequences:
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            _, h = ae.encoder(x)
            latents.append(h[-1].squeeze(0).cpu().numpy())

    latents = np.array(latents)

    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(latents)

    plt.figure(figsize=(8,6))
    plt.scatter(
        z_2d[:,0], z_2d[:,1],
        c=np.arange(len(z_2d)), cmap="viridis", alpha=0.7
    )
    plt.colorbar(label="Índice de secuencia")
    plt.title("Espacio latente – Secuencias generadas")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.grid(True)
    plt.show()


# ============================================================
# 💾 EXPORTAR CSV FINAL
# ============================================================

def export_csv(seqs, preds, feature_columns):
    rows = []
    for s, p in zip(seqs, preds):
        rows.append(np.concatenate([s[-1], p]))

    df = pd.DataFrame(
        rows,
        columns=feature_columns + OUTPUT_COLUMNS
    )
    df.to_csv(CSV_FOLDER+OUTPUT_CSV, index=False)
    print(f"✅ CSV generado: {CSV_FOLDER+OUTPUT_CSV}")

# ============================================================
# 🏁 MAIN
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")

    X_raw, Y_raw, categorical_info, feature_columns  = cargar_csv_onehot(
    ruta_csv=CSV_FOLDER + DATASET_PATH,
    columnas_target=OUTPUT_COLUMNS
    )

    input_size = X_raw.shape[1]

    dataset = RealDataset(X_raw, Y_raw)
    X_real, Y_real = dataset.create_windows()

    # 1️⃣ Autoencoder → generar inputs nuevos
    ae = train_autoencoder(X_real, input_size, device)
    
    categorical_indices = get_categorical_indices(
    categorical_info,
    feature_columns
    )

    X_synth = generate_sequences(ae, X_real, device)

    X_synth_discrete = discretize_categoricals(
        X_synth,
        categorical_indices
    )

    # 2️⃣ GRU entrenado SOLO con reales
    gru = train_gru_real(X_real, Y_real, input_size, device)

    # 3️⃣ Predicción emociones sintéticas
    with torch.no_grad():
        preds = gru(torch.tensor(X_synth_discrete, dtype=torch.float32).to(device)).cpu().numpy()

    # 4️⃣ Exportar CSV final
    export_csv(X_synth_discrete, preds, feature_columns)

    # 5️⃣ Visualizar espacio latente de secuencias nuevas
    plot_latents(ae, X_synth_discrete, device)
