import os
import pandas as pd
import pydicom

dataset_path = r"C:\Users\lacostea\.cache\kagglehub\datasets\trainingdatapro\computed-tomography-ct-of-the-brain\versions\1"
csv_file = os.path.join(dataset_path, "ct_brain.csv")
files_dir = os.path.join(dataset_path, "files")

# --- 1. Inspecter le CSV ---
df = pd.read_csv(csv_file)
print("=== Colonnes CSV ===")
print(df.columns)
print("\n=== Aperçu CSV ===")
print(df.head())

print(f"\nNombre total de scans dans le CSV : {len(df)}")
print(f"Types de maladies présents : {df['type'].unique()}")

# --- 2. Vérifier la structure des dossiers et les fichiers ---
print("\n=== Structure des dossiers ===")
for disease in os.listdir(files_dir):
    disease_path = os.path.join(files_dir, disease)
    if os.path.isdir(disease_path):
        files = os.listdir(disease_path)
        dcm_files = [f for f in files if f.endswith(".dcm")]
        jpg_files = [f for f in files if f.endswith(".jpg")]
        print(f"{disease}: {len(dcm_files)} DICOM, {len(jpg_files)} JPEG")

# --- 3. Vérifier le contenu d'un DICOM ---
# Choisir un fichier DICOM exemple
for root, dirs, files in os.walk(files_dir):
    for f in files:
        if f.endswith(".dcm"):
            dcm_path = os.path.join(root, f)
            dcm = pydicom.dcmread(dcm_path)
            print(f"\nExemple DICOM : {dcm_path}")
            print(f"Dimensions pixel array : {dcm.pixel_array.shape}")
            print(f"Patient ID (si présent) : {getattr(dcm, 'PatientID', 'N/A')}")
            print(f"Modality : {getattr(dcm, 'Modality', 'N/A')}")
            # On s'arrête après le premier fichier
            break
    break
