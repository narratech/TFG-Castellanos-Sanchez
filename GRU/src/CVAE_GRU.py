import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# ======================
# CONFIG
# ======================
SEQ_LEN = 35
INPUT_DIM = 3        # <-- AJUSTA ESTO
Y_DIM = 4            # <-- nº emociones
LATENT_DIM = 16
BATCH_SIZE = 64
EPOCHS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_CSV = "data.csv"
OUTPUT_CSV = "synthetic_data.csv"

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(INPUT_CSV)

X = df.iloc[:, :-Y_DIM].values
y = df.iloc[:, -Y_DIM:].values

# reshape a secuencia
X = X.reshape(-1, SEQ_LEN, INPUT_DIM)

# normalizar X
scaler = MinMaxScaler()
X_flat = X.reshape(-1, INPUT_DIM)
X_flat = scaler.fit_transform(X_flat)
X = X_flat.reshape(-1, SEQ_LEN, INPUT_DIM)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ======================
# MODEL
# ======================
class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder_gru = nn.GRU(INPUT_DIM + Y_DIM, 64, batch_first=True)
        self.fc_mu = nn.Linear(64, LATENT_DIM)
        self.fc_logvar = nn.Linear(64, LATENT_DIM)

        self.decoder_gru = nn.GRU(LATENT_DIM + Y_DIM, 64, batch_first=True)
        self.output_layer = nn.Linear(64, INPUT_DIM)

    def encode(self, x, y):
        y_expanded = y.unsqueeze(1).repeat(1, SEQ_LEN, 1)
        inp = torch.cat([x, y_expanded], dim=-1)
        _, h = self.encoder_gru(inp)
        h = h.squeeze(0)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        z_expanded = z.unsqueeze(1).repeat(1, SEQ_LEN, 1)
        y_expanded = y.unsqueeze(1).repeat(1, SEQ_LEN, 1)
        inp = torch.cat([z_expanded, y_expanded], dim=-1)
        out, _ = self.decoder_gru(inp)
        return self.output_layer(out)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparametrize(mu, logvar)
        x_recon = self.decode(z, y)
        return x_recon, mu, logvar

model = CVAE().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ======================
# LOSS
# ======================
def loss_fn(x_recon, x, mu, logvar):
    recon = nn.MSELoss()(x_recon, x)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + 0.001 * kld

# ======================
# TRAIN
# ======================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        
        optimizer.zero_grad()
        x_recon, mu, logvar = model(xb, yb)
        loss = loss_fn(x_recon, xb, mu, logvar)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ======================
# GENERATE NEW DATA
# ======================
model.eval()

NUM_NEW_SAMPLES = len(X)  # puedes cambiar esto

# sample y reales (mejor distribución)
y_sample = y_tensor[np.random.choice(len(y_tensor), NUM_NEW_SAMPLES)]

z = torch.randn(NUM_NEW_SAMPLES, LATENT_DIM).to(DEVICE)
y_sample = y_sample.to(DEVICE)

with torch.no_grad():
    X_fake = model.decode(z, y_sample)

X_fake = X_fake.cpu().numpy()
y_sample = y_sample.cpu().numpy()

# desnormalizar X
X_fake_flat = X_fake.reshape(-1, INPUT_DIM)
X_fake_flat = scaler.inverse_transform(X_fake_flat)
X_fake = X_fake_flat.reshape(NUM_NEW_SAMPLES, -1)

# combinar con y
synthetic = np.concatenate([X_fake, y_sample], axis=1)

# guardar CSV
columns = list(df.columns)
synthetic_df = pd.DataFrame(synthetic, columns=columns)
synthetic_df.to_csv(OUTPUT_CSV, index=False)

print("✅ Dataset sintético guardado en:", OUTPUT_CSV)