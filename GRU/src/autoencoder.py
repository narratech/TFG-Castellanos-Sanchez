import argparse

parser = argparse.ArgumentParser(description="Script para entrenar GRU")
parser.add_argument("--csv", type=str, required=True, help="Ruta del archivo CSV de entrada")
parser.add_argument("--epochs", type=int, default=100, help="Número de épocas")

args = parser.parse_args()

# ============================================================
# 🔧 CONFIGURACIÓN
# ============================================================


DATASET_PATH = args.csv         # real (9 + 6)
OUTPUT_CSV = "generated_"
CSV_FOLDER = "datatest/"

OUTPUT_SIZE = 6
SEQUENCE_LENGTH = 35

LATENT_SIZE = 32
HIDDEN_SIZE = 64

AE_EPOCHS = 160
GRU_EPOCHS = 100
BATCH_SIZE = 32
LR = 0.01
ACCURACY_THRESHOLD = 0.1

N_SYNTHETIC = args.epochs
LATENT_NOISE_STD = 0.1

USE_CUDA = True

EMOTIONS = ["Ira", "Miedo", "Felicidad", "Tristeza", "Sorpresa", "Disgusto"]

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
    optimizer = torch.optim.Adam(ae.parameters(), lr=LR)
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

def argmax(X_synth):
    # Hacemos una copia para no modificar la original
    X_synth_discrete = X_synth.copy()

    for var, cols in categorical_info.items():
        # Obtener los índices de las columnas
        col_indices = [i for i, c in enumerate(cols)]  # si los tienes como nombres, mapéalos a índices
        
        # Tomamos solo las columnas de la variable categórica
        cat_vals = X_synth[:, :, col_indices]  # shape: (N, SEQ_LEN, n_categories)
        
        # Aplicamos argmax a lo largo de la dimensión de categorías (axis=2)
        argmax_vals = np.argmax(cat_vals, axis=2)  # shape: (N, SEQ_LEN)
        
        # Reconstruir One-Hot: limpiar las columnas originales
        X_synth_discrete[:, :, col_indices] = 0
        
        # Poner 1 en la categoría seleccionada
        N, SEQ_LEN = argmax_vals.shape
        for i in range(N):
            for t in range(SEQ_LEN):
                X_synth_discrete[i, t, col_indices[argmax_vals[i, t]]] = 1
        
    return X_synth_discrete

def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()  # <- CORRECCIÓN
            y = y.to(device).float()  # <- CORRECCIÓN
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

# ============================================================
# 🚀 ENTRENAR GRU (SOLO DATOS REALES)
# ============================================================

def train_gru_real(X, Y, input_size, device):
    model = GRUEmotion(input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
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
            latents.append(h[-1].squeeze(0).cpu().numpy())  # <--- OJO squeeze aquí

    latents = np.array(latents)  # forma (N, LATENT_SIZE)

    # PCA 2D
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

def export_csv(seqs, preds, input_size):
    rows = []
    for s, p in zip(seqs, preds):
        rows.append(np.concatenate([s[-1], p]))

    df = pd.DataFrame(
        rows,
        columns=[f"Input_{i+1}" for i in range(input_size)] + EMOTIONS
    )
    df.to_csv(CSV_FOLDER+OUTPUT_CSV+DATASET_PATH, index=False)
    print(f"✅ CSV generado: {CSV_FOLDER+OUTPUT_CSV+DATASET_PATH}")

# ============================================================
# 🏁 MAIN
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")

    targets = [
    "Ira",
    "Miedo",
    "Felicidad",
    "Tristeza",
    "Sorpresa",
    "Disgusto"
    ]

    X_raw, Y_raw, categorical_info = cargar_csv_onehot(
    ruta_csv=CSV_FOLDER + DATASET_PATH,
    columnas_target=targets
    )

    input_size = X_raw.shape[1]

    dataset = RealDataset(X_raw, Y_raw)
    X_real, Y_real = dataset.create_windows()

    # 1️⃣ Autoencoder → generar inputs nuevos
    ae = train_autoencoder(X_real, input_size, device)
    X_synth = generate_sequences(ae, X_real, device)

    X_synth_discrete = argmax(X_synth)

    # 2️⃣ GRU entrenado SOLO con reales
    gru = train_gru_real(X_real, Y_real, input_size, device)

    # 3️⃣ Predicción emociones sintéticas
    with torch.no_grad():
        preds = gru(torch.tensor(X_synth_discrete, dtype=torch.float32).to(device)).cpu().numpy()

    # 4️⃣ Exportar CSV final
    export_csv(X_synth_discrete, preds, input_size)

    # 5️⃣ Visualizar espacio latente de secuencias nuevas
    plot_latents(ae, X_synth_discrete, device)
