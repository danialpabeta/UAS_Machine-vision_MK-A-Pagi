import pandas as pd
import torch
import numpy as np
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load dataset
train_data_path = '/home/barelang/Downloads/archive(1)/emnist-bymerge-train.csv'
val_data_path = '/home/barelang/Downloads/archive(1)/emnist-bymerge-test.csv'
data_train = pd.read_csv(train_data_path, header=None, nrows=3000)
data_val = pd.read_csv(val_data_path, header=None, nrows=3000)
print("Dataset loaded successfully.")

# Dataset class
def preprocess_image(data):
    data = np.clip(data, 0, 255).astype(np.uint8).reshape(28, 28)
    return Image.fromarray(data).convert("RGB")

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        label = self.dataframe.iloc[idx, 0]
        img_data = self.dataframe.iloc[idx, 1:].values
        image = preprocess_image(img_data)
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load datasets
train_dataset = CustomDataset(data_train, transform=transform)
val_dataset = CustomDataset(data_val, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Initialize model
model = models.alexnet(pretrained=True)
model.classifier[6] = nn.Linear(4096, 200)
for param in model.features.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
device = torch.device('cuda')
model.to(device)

# Training loop
num_epochs = 15
train_losses, train_accuracies = [], []
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score

# Leave-One-Out Cross Validation (LOOCV) Implementation
loo = LeaveOneOut()
data_array = data_train.to_numpy()
all_preds, all_labels = [], []
# Variabel untuk menyimpan metrik per iterasi
accuracies = []
precisions = []

print("Starting LOOCV...")

for train_idx, test_idx in tqdm(loo.split(data_array)):
    train_samples = data_array[train_idx]
    test_sample = data_array[test_idx]

    # Create datasets and dataloaders
    train_dataset = CustomDataset(pd.DataFrame(train_samples), transform=transform)
    test_dataset = CustomDataset(pd.DataFrame(test_sample), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Reinitialize model and optimizer for each LOOCV iteration
    model = models.alexnet(pretrained=True)
    model.classifier[6] = nn.Linear(4096, 200)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train on train_loader
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Test on test_loader
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_preds.append(torch.argmax(outputs, dim=1).cpu().item())
            all_labels.append(labels.cpu().item())

# Calculate Evaluation Metrics
conf_matrix = confusion_matrix(all_labels, all_preds)
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='macro')

print("\nEvaluation Results:")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-Score: {f1:.4f}")