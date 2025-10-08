# analyze_training_stats.py
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import seaborn as sns

# -----------------------------
# Arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Analyse des statistiques d'entraînement (Excel)")
parser.add_argument("--file", type=str, default="C:\\Users\\lacostea\\PycharmProjects\\PythonProject1\\SRC\\stats_allruns_37datasets_20251008_030325.xlsx", help="Chemin du fichier Excel de stats")
parser.add_argument("--top", type=int, default=10, help="Nombre de datasets à afficher dans les classements")
args = parser.parse_args()

excel_path = Path(args.file)
if not excel_path.exists():
    raise FileNotFoundError(f"❌ Fichier non trouvé : {excel_path}")

# -----------------------------
# Chargement
# -----------------------------
print(f"📂 Chargement du fichier Excel : {excel_path.name}")
xls = pd.ExcelFile(excel_path)
df_all = pd.read_excel(xls, "All_Results")

# -----------------------------
# Statistiques globales
# -----------------------------
print("\n=== 📈 Statistiques globales ===")

# Moyenne et écart-type sur toutes les runs
df_stats = (
    df_all.groupby("dataset")
    .agg({
        "val_acc": ["mean", "std", "max"],
        "val_loss": ["mean", "std", "min"],
        "train_loss": ["mean", "std"],
    })
)
df_stats.columns = ['_'.join(col) for col in df_stats.columns]
df_stats = df_stats.reset_index().sort_values("val_acc_mean", ascending=False)

print(df_stats.head(args.top))

# -----------------------------
# Sauvegarde d’un résumé
# -----------------------------
summary_path = excel_path.parent / f"summary_{excel_path.stem}.xlsx"
with pd.ExcelWriter(summary_path) as writer:
    df_stats.to_excel(writer, sheet_name="Summary_Sorted", index=False)
    df_all.to_excel(writer, sheet_name="Raw_Data", index=False)

print(f"\n✅ Résumé sauvegardé dans : {summary_path}")

# -----------------------------
# TOP datasets
# -----------------------------
top_n = args.top
top_datasets = df_stats.head(top_n)["dataset"].tolist()

plt.figure(figsize=(10, 6))
sns.barplot(data=df_stats.head(top_n), x="dataset", y="val_acc_mean")
plt.xticks(rotation=45, ha='right')
plt.title(f"Top {top_n} prétraitements - Moyenne Validation Accuracy")
plt.ylabel("Validation Accuracy moyenne")
plt.xlabel("Dataset")
plt.tight_layout()
plt.show()

# -----------------------------
# Variabilité (écart-type)
# -----------------------------
plt.figure(figsize=(10, 6))
sns.barplot(data=df_stats.head(top_n), x="dataset", y="val_acc_std")
plt.xticks(rotation=45, ha='right')
plt.title(f"Variabilité de la performance (std) - Top {top_n}")
plt.ylabel("Écart-type Accuracy")
plt.xlabel("Dataset")
plt.tight_layout()
plt.show()

# -----------------------------
# Comparaison Accuracy / Loss
# -----------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_stats, x="val_loss_mean", y="val_acc_mean")
plt.title("Relation entre perte moyenne et précision moyenne")
plt.xlabel("Validation Loss moyenne")
plt.ylabel("Validation Accuracy moyenne")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# -----------------------------
# Zoom sur un dataset précis
# -----------------------------
choice = input("\n🔍 Entrer un nom de dataset à visualiser (ou laisser vide pour ignorer) : ").strip()
if choice:
    if choice not in df_all["dataset"].unique():
        print(f"❌ Dataset '{choice}' introuvable.")
    else:
        df_choice = df_all[df_all["dataset"] == choice]
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=df_choice, x="epoch", y="val_acc", hue="run", marker="o")
        plt.title(f"Évolution Validation Accuracy - {choice}")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(title="Run")
        plt.tight_layout()
        plt.show()

print("\n🎯 Analyse terminée.")
