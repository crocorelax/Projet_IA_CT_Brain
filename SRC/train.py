# train_compare_37datasets_excel.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from CNN import SimpleCNN
from Prepare_Dataset import dataloaders  # contient les DataLoaders

# -------------------------
# Arguments ligne de commande
# -------------------------
parser = argparse.ArgumentParser(description="Compare prétraitements sur plusieurs runs")
parser.add_argument("--n_runs", type=int, default=40, help="Nombre de répétitions d'entraînement")
parser.add_argument("--epochs", type=int, default=5, help="Nombre d'époques")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
args = parser.parse_args()

n_runs = args.n_runs
num_epochs = args.epochs
learning_rate = args.lr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Device utilisé : {device}")
print(f"📦 Nombre de datasets : {len(dataloaders)}")

# -------------------------
# Fonction d'entraînement pour un epoch
# -------------------------
def train_one_epoch(model, loader_train, loader_val, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader_train:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(loader_train)

    # Validation
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader_val:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_loss /= len(loader_val)
    val_acc = correct / total
    return train_loss, val_loss, val_acc

# -------------------------
# Fonction d'entraînement sur un dataset complet
# -------------------------
def train_on_dataset(name, dataloader, n_runs, num_epochs, lr):
    results = []
    print(f"\n=== 🔧 Entraînement sur dataset : {name} ===")

    for run in range(1, n_runs + 1):
        print(f"\n---- Run {run}/{n_runs} ----")

        model = SimpleCNN(num_classes=3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, num_epochs + 1):
            train_loss, val_loss, val_acc = train_one_epoch(
                model, dataloader, dataloader, optimizer, device
            )

            results.append({
                "dataset": name,
                "run": run,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc
            })

            print(f"Epoch {epoch:02d}/{num_epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.4f}")

    return pd.DataFrame(results)


# -------------------------
# Entraînement sur tous les datasets
# -------------------------
all_results = []
for name, loader in dataloaders.items():
    df_dataset = train_on_dataset(name, loader, n_runs, num_epochs, learning_rate)
    all_results.append(df_dataset)

# Fusionner tous les résultats
df_all = pd.concat(all_results, ignore_index=True)

# -------------------------
# Organisation du Excel
# -------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
excel_filename = f"stats_allruns_37datasets_{timestamp}.xlsx"

with pd.ExcelWriter(excel_filename) as writer:
    # Feuille récap globale
    df_all.to_excel(writer, sheet_name="All_Results", index=False)

     # Moyennes par dataset
    df_summary = (
        df_all.groupby("dataset")[["val_acc", "val_loss", "train_loss"]]
        .mean()
        .sort_values("val_acc", ascending=False)
    )
    df_summary.to_excel(writer, sheet_name="Summary")

    # Feuille individuelle pour chaque dataset
    for name, df_subset in df_all.groupby("dataset"):
        df_subset.to_excel(writer, sheet_name=name[:31], index=False)  # Excel limite à 31 caractères

print(f"\n✅ Tous les résultats sauvegardés dans : {excel_filename}")

# -------------------------
# Graphiques : top 5 datasets
# -------------------------
df_summary_top = df_summary.head(5)
print("\n🏆 Top 5 des prétraitements :")
print(df_summary_top)

plt.figure(figsize=(10, 6))
plt.bar(df_summary_top.index, df_summary_top["val_acc"])
plt.xticks(rotation=45, ha='right')
plt.title("Top 5 des prétraitements par précision moyenne (validation)")
plt.ylabel("Validation Accuracy moyenne")
plt.tight_layout()
plt.show()
