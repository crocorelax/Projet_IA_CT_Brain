import pandas as pd
from scipy.stats import f_oneway, kruskal, shapiro, levene

# -------------------------
# Charger le résumé des 37 datasets
# -------------------------
excel_file = "summary_stats_allruns_37datasets_20251008_030325.xlsx"
df_summary = pd.read_excel(excel_file, sheet_name="Summary_Sorted")

# -------------------------
# Extraire les paramètres depuis le nom du dataset
# -------------------------
def parse_dataset_name(name):
    if name == "brut":
        return {"percentile": "brut", "kernel": 0, "min_area": 0, "blur": False}
    parts = name.split("_")
    return {
        "percentile": parts[0],
        "kernel": int(parts[1][1:]) if parts[1].startswith("k") else 0,
        "min_area": int(parts[2][1:]) if parts[2].startswith("a") else 0,
        "blur": parts[3][-1] == "1" if len(parts) > 3 else False
    }

params = df_summary["dataset"].apply(parse_dataset_name)
df_params = pd.DataFrame(list(params))
df_summary = pd.concat([df_summary, df_params], axis=1)

# -------------------------
# Fonction de test statistique automatique
# -------------------------
def test_effect(df, param, metric):
    # Créer les groupes selon le paramètre
    groups = [df[df[param] == v][metric] for v in df[param].unique()]
    
    # Vérifier normalité de chaque groupe (Shapiro-Wilk)
    normal = all(len(g) >= 3 and shapiro(g).pvalue > 0.05 for g in groups)
    
    # Choix du test
    if normal:
        f_val, p_val = f_oneway(*groups)
        test_name = "ANOVA"
    else:
        f_val, p_val = kruskal(*groups)
        test_name = "Kruskal-Wallis"
    
    signif = "oui" if p_val < 0.05 else "non"
    print(f"{param}: {test_name}, F/H={f_val:.6f}, p={p_val:.12f} → effet significatif ? {signif}")

# -------------------------
# Tester tous les paramètres pour toutes les métriques
# -------------------------
metrics = {
    "Vitesse de généralisation": "val_loss_mean",
    "Robustesse": "val_acc_std"
}

for metric_name, metric_col in metrics.items():
    print(f"\n=== {metric_name} ===")
    for param in ["percentile", "kernel", "min_area", "blur"]:
        test_effect(df_summary, param, metric_col)

# -------------------------
# Valeurs uniques et comptage
# -------------------------
print("\nDistribution des paramètres:")
print("blur:\n", df_summary["blur"].value_counts())
print("percentile:\n", df_summary["percentile"].value_counts())
print("kernel:\n", df_summary["kernel"].value_counts())
print("\nNombre d'échantillons par kernel pour val_loss_mean:")
print(df_summary.groupby("kernel")["val_loss_mean"].count())

# -------------------------
# Exemple de tests complémentaires pour 'kernel'
# -------------------------
kernel_groups = [df_summary[df_summary["kernel"] == v]["val_loss_mean"] for v in df_summary["kernel"].unique()]

# Normalité
for i, g in enumerate(kernel_groups):
    stat, p = shapiro(g)
    print(f"kernel {i+1} Shapiro: statistic={stat:.3f}, pvalue={p:.4f}")

# Homogénéité des variances
stat, p = levene(*kernel_groups)
print(f"Levene test: statistic={stat:.3f}, pvalue={p:.4f}")
