# eval.py
import torch
from CNN import SimpleCNN
from Prepare_Dataset import test_loader
from sklearn.metrics import classification_report

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le modèle
model = SimpleCNN(num_classes=3).to(device)
model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
model.eval()

# Préparer liste des labels
label_names = list(test_loader.dataset.label_map.keys())

# Évaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Rapport classification
print("Classification Report :")
print(classification_report(all_labels, all_preds, target_names=label_names))
