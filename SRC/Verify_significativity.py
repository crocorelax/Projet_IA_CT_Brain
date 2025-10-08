import pandas as pd
from scipy.stats import f_oneway

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
# Fonction test ANOVA
# -------------------------
def test_anova(df, param, metric):
    groups = [df[df[param] == v][metric] for v in df[param].unique()]
    f_val, p_val = f_oneway(*groups)
    return f_val, p_val

# -------------------------
# Tester chaque paramètre pour vitesse et robustesse
# -------------------------
metrics = {
    "Vitesse de généralisation": "val_loss_mean",
    "Robustesse": "val_acc_std"
}

for metric_name, metric_col in metrics.items():
    print(f"\n=== {metric_name} ===")
    for param in ["percentile", "kernel", "min_area", "blur"]:
        f_val, p_val = test_anova(df_summary, param, metric_col)
        signif = "oui" if p_val < 0.05 else "non"
        print(f"{param}: F={f_val:.3f}, p={p_val:.4f} → effet significatif ? {signif}")

df_summary["blur"].value_counts()
df_summary["percentile"].value_counts()
df_summary["kernel"].value_counts()
df_summary.groupby("kernel")["val_loss_mean"].count()

from scipy.stats import shapiro, levene

# Exemple pour "kernel"
groups = [df_summary[df_summary["kernel"] == v]["val_loss_mean"] for v in df_summary["kernel"].unique()]

# Normalité des groupes
for i, g in enumerate(groups):
    print(f"kernel {i+1} Shapiro:", shapiro(g))

# Homogénéité des variances
print("Levene test:", levene(*groups))
