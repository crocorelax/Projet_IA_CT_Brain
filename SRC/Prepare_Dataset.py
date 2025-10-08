import os
import pandas as pd
import numpy as np
import pydicom
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from sklearn.model_selection import train_test_split
from itertools import product

# -------------------------
# Paths
# -------------------------
dataset_path = r"C:\Users\lacostea\.cache\kagglehub\datasets\trainingdatapro\computed-tomography-ct-of-the-brain\versions\1"
csv_file = os.path.join(dataset_path, "ct_brain.csv")
img_dir = os.path.join(dataset_path, "files")

df = pd.read_csv(csv_file)

# -------------------------
# Label mapping
# -------------------------
label_map = {label: idx for idx, label in enumerate(df['type'].unique())}

# -------------------------
# PyTorch transforms
# -------------------------
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# -------------------------
# Dataset Class
# -------------------------
class BrainCTDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, label_map=None,
                 percentiles=(5,95), kernel_size=3, min_area=0, blur=True,
                 mode_name="custom", output_size=(256,256)):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = label_map
        self.percentiles = percentiles
        self.kernel_size = kernel_size
        self.min_area = min_area
        self.blur = blur
        self.output_size = output_size
        self.mode_name = mode_name

    def __len__(self):
        return len(self.df)

    def load_dcm(self, path):
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)
        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            img = img * dcm.RescaleSlope + dcm.RescaleIntercept
        return img

    def preprocess_img(self, img):
        # --- bornes ---
        lower, upper = np.percentile(img, self.percentiles[0]), np.percentile(img, self.percentiles[1])
        img_clipped = np.clip(img, lower, upper)

        # --- masque binaire ---
        mask = ((img_clipped > lower) & (img_clipped < upper)).astype(np.uint8) * 255
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # --- composantes connectées ---
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_label).astype(np.uint8) * 255

        # --- suppression petits composants ---
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.min_area:
                mask[labels == i] = 0

        # --- flou optionnel ---
        mask_float = mask / 255.0
        if self.blur:
            mask_float = cv2.GaussianBlur(mask_float, (3, 3), 0)

        # --- application masque ---
        img_masked = img_clipped * mask_float

        # --- normalisation ---
        img_masked -= img_masked.min()
        if img_masked.max() > 0:
            img_masked /= img_masked.max()

        # --- redimensionnement ---
        img_resized = cv2.resize(img_masked, self.output_size, interpolation=cv2.INTER_LINEAR)
        return img_resized.astype(np.float32)

    def __getitem__(self, idx):
        file_name = self.df.iloc[idx]['jpg'].strip("/").replace(".jpg", ".dcm")
        img_path = os.path.join(self.img_dir, file_name)
        img = self.load_dcm(img_path)

        if self.mode_name == "brut":
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        else:
            img = self.preprocess_img(img)

        img_uint8 = (img * 255).astype(np.uint8)
        image = Image.fromarray(img_uint8).convert("RGB")

        label = self.label_map[self.df.iloc[idx]['type']]
        if self.transform:
            image = self.transform(image)

        return image, label


# -------------------------
# Split train/val/test
# -------------------------
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['type'], random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, stratify=test_df['type'], random_state=42)

# -------------------------
# Génération des configurations (compatible avec parse_dataset_name)
# -------------------------
from itertools import product, combinations
area_opts = [0, 500, 1000]

configs = [("brut", None)]  # Mode brut de base

percentile_opts = [(5, 95), (10, 90)]
kernel_opts = [3, 5, 9]
blur_opts = [False, True]

param_names = ["percentile", "kernel", "min_area", "blur"]

for r in range(1, len(param_names) + 1):
    for subset in combinations(param_names, r):
        # Génération de toutes les combinaisons de valeurs
        param_values = []
        for name in subset:
            if name == "percentile":
                param_values.append(percentile_opts)
            elif name == "kernel":
                param_values.append(kernel_opts)
            elif name == "min_area":
                param_values.append(area_opts)
            elif name == "blur":
                param_values.append(blur_opts)

        for values in product(*param_values):
            params_dict = dict(zip(subset, values))

            # Valeurs par défaut pour les manquants
            p_val = params_dict.get("percentile", "none")
            k_val = params_dict.get("kernel", 0)
            a_val = params_dict.get("min_area", 0)
            b_val = params_dict.get("blur", False)

            # Nom compatible avec ton parser
            if p_val == "none":
                name = f"none_k{k_val}_a{a_val}_blur{int(b_val)}"
            else:
                name = f"p{p_val[0]}-{p_val[1]}_k{k_val}_a{a_val}_blur{int(b_val)}"

            configs.append((
                name,
                (
                    p_val if p_val != "none" else (5, 95),  # valeur neutre interne
                    k_val,
                    a_val,
                    b_val
                )
            ))

print(f"🧩 {len(configs)} configurations générées (y compris 'brut').")

# -------------------------
# Création des Datasets et Dataloaders
# -------------------------
batch_size = 8
datasets = {}
dataloaders = {}

for name, params in configs:
    if params is None:
        ds = BrainCTDataset(train_df, img_dir, transform=train_transform,
                            label_map=label_map, mode_name="brut")
    else:
        percentiles, kernel, min_area, blur = params
        ds = BrainCTDataset(train_df, img_dir, transform=train_transform,
                            label_map=label_map,
                            percentiles=percentiles, kernel_size=kernel,
                            min_area=min_area, blur=blur, mode_name=name)
    datasets[name] = ds
    dataloaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=True)

print(f"✅ {len(dataloaders)} dataloaders créés (y compris 'brut').")
