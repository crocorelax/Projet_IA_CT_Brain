import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Chargement des résultats
# -------------------------
excel_file = "stats_allruns_37datasets_20251008_030325.xlsx"
df_all = pd.read_excel(excel_file, sheet_name="All_Results")

# -------------------------
# Calcul de métriques dérivées
# -------------------------
summary_rows = []

for dataset in df_all['dataset'].unique():
    df_ds = df_all[df_all['dataset'] == dataset]

    # Pour chaque run, extraire la vitesse (epoch où val_loss < threshold)
    run_metrics = []
    for run in df_ds['run'].unique():
        df_run = df_ds[df_ds['run'] == run].sort_values('epoch')

        # Vitesse : epoch où val_loss < 0.2 ou lowest val_loss
        val_loss_series = df_run['val_loss'].values
        min_val_loss = val_loss_series.min()
        min_epoch = df_run['epoch'].values[val_loss_series.argmin()]

        # Robustesse : écart type sur les runs
        run_metrics.append({
            'min_val_loss': min_val_loss,
            'min_epoch': min_epoch,
            'final_val_loss': val_loss_series[-1],
            'final_val_acc': df_run['val_acc'].values[-1]
        })

    # Moyenne et std sur les runs
    min_val_loss_mean = np.mean([m['min_val_loss'] for m in run_metrics])
    min_val_loss_std = np.std([m['min_val_loss'] for m in run_metrics])
    min_epoch_mean = np.mean([m['min_epoch'] for m in run_metrics])
    min_epoch_std = np.std([m['min_epoch'] for m in run_metrics])
    final_val_loss_mean = np.mean([m['final_val_loss'] for m in run_metrics])
    final_val_loss_std = np.std([m['final_val_loss'] for m in run_metrics])
    final_val_acc_mean = np.mean([m['final_val_acc'] for m in run_metrics])
    final_val_acc_std = np.std([m['final_val_acc'] for m in run_metrics])

    summary_rows.append({
        'dataset': dataset,
        'min_val_loss_mean': min_val_loss_mean,
        'min_val_loss_std': min_val_loss_std,
        'min_epoch_mean': min_epoch_mean,
        'min_epoch_std': min_epoch_std,
        'final_val_loss_mean': final_val_loss_mean,
        'final_val_loss_std': final_val_loss_std,
        'final_val_acc_mean': final_val_acc_mean,
        'final_val_acc_std': final_val_acc_std
    })

df_summary = pd.DataFrame(summary_rows).sort_values('final_val_acc_mean', ascending=False)

# -------------------------
# Visualisation rapide
# -------------------------
plt.figure(figsize=(12,6))
plt.errorbar(df_summary['dataset'], df_summary['final_val_acc_mean'],
             yerr=df_summary['final_val_acc_std'], fmt='o', capsize=5)
plt.xticks(rotation=90)
plt.ylabel("Validation Accuracy (mean ± std)")
plt.title("Comparaison des IA : Robustesse et Performance finale")
plt.show()

plt.figure(figsize=(12,6))
plt.errorbar(df_summary['dataset'], df_summary['min_epoch_mean'],
             yerr=df_summary['min_epoch_std'], fmt='o', capsize=5)
plt.xticks(rotation=90)
plt.ylabel("Epoch de min val_loss (vitesse d'apprentissage)")
plt.title("Vitesse d'apprentissage : quelle IA converge le plus vite ?")
plt.show()

plt.figure(figsize=(12,6))
plt.errorbar(df_summary['dataset'], df_summary['min_val_loss_mean'],
             yerr=df_summary['min_val_loss_std'], fmt='o', capsize=5)
plt.xticks(rotation=90)
plt.ylabel("Min Validation Loss (mean ± std)")
plt.title("Robustesse : variance de val_loss minimale")
plt.show()

# -------------------------
# Sauvegarde
# -------------------------
df_summary.to_excel("summary_analysis_speed_robustness.xlsx", index=False)
print("Résumé sauvegardé dans summary_analysis_speed_robustness.xlsx")
