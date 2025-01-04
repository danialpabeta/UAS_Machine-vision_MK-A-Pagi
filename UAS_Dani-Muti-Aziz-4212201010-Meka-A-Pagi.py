# Import libraries for data processing, model building, and evaluation
import pandas as pd
import torch
import numpy as np
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from PIL import Image
from tqdm import tqdm

# Load dataset
train_data_path = '/home/barelang/Downloads/archive(1)/emnist-bymerge-train.csv'
val_data_path = '/home/barelang/Downloads/archive(1)/emnist-bymerge-test.csv'

# Load a subset of 3000 samples for training and validation for faster processing
data_train = pd.read_csv(train_data_path, header=None, nrows=3000)
data_val = pd.read_csv(val_data_path, header=None, nrows=3000)
print("Dataset loaded successfully.")

# Preprocessing function for raw pixel data
def preprocess_image(data):
    """
    Converts raw image data into a 28x28 image, clips values between 0-255, 
    and converts it to an RGB PIL image.
    """
    data = np.clip(data, 0, 255).astype(np.uint8).reshape(28, 28)
    return Image.fromarray(data).convert("RGB")

# Custom dataset class to handle EMNIST data
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Initialize the dataset with a DataFrame and optional image transformations.
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieves the image and label for a given index, applies preprocessing and transformations.
        """
        label = self.dataframe.iloc[idx, 0]  # First column contains labels
        img_data = self.dataframe.iloc[idx, 1:].values  # Remaining columns contain pixel data
        image = preprocess_image(img_data)  # Preprocess image data
        if self.transform:
            image = self.transform(image)  # Apply transformations (if any)
        return image, label

# Transformations for input data to be compatible with AlexNet (224x224, tensor format)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize 28x28 to 224x224
    transforms.ToTensor()  # Convert to PyTorch tensor
])

# Create datasets and data loaders for training and validation
train_dataset = CustomDataset(data_train, transform=transform)
val_dataset = CustomDataset(data_val, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Training data loader
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)  # Validation data loader

# Initialize a pretrained AlexNet model for transfer learning
model = models.alexnet(pretrained=True)  # Load pretrained AlexNet
model.classifier[6] = nn.Linear(4096, 200)  # Replace the final layer with 200 output classes

# Freeze feature extraction layers to only train the classifier
for param in model.features.parameters():
    param.requires_grad = False

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adam optimizer with learning rate 0.0001

# Configure the device for GPU acceleration (if available)
device = torch.device('cuda')
model.to(device)  # Move model to GPU

# Leave-One-Out Cross Validation (LOOCV)
from sklearn.model_selection import LeaveOneOut

# Convert the training data into a NumPy array for LOOCV
data_array = data_train.to_numpy()

# Initialize lists to store predictions and labels for evaluation
all_preds, all_labels = [], []

print("Starting LOOCV...")

# Leave-One-Out Cross Validation implementation
loo = LeaveOneOut()
for train_idx, test_idx in tqdm(loo.split(data_array)):  # Iterate over each split
    # Split data into training and test sets for this fold
    train_samples = data_array[train_idx]
    test_sample = data_array[test_idx]

    # Create datasets and dataloaders for the current LOOCV split
    train_dataset = CustomDataset(pd.DataFrame(train_samples), transform=transform)
    test_dataset = CustomDataset(pd.DataFrame(test_sample), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Reinitialize the model and optimizer for each LOOCV iteration
    model = models.alexnet(pretrained=True)
    model.classifier[6] = nn.Linear(4096, 200)  # Update the classifier for 200 classes
    model.to(device)

    criterion = nn.CrossEntropyLoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Optimizer

    # Training loop for the current fold
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()  # Reset gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

    # Validation loop for the current fold
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            outputs = model(inputs)  # Forward pass
            all_preds.append(torch.argmax(outputs, dim=1).cpu().item())  # Store predictions
            all_labels.append(labels.cpu().item())  # Store true labels

# Calculate evaluation metrics
conf_matrix = confusion_matrix(all_labels, all_preds)  # Confusion matrix
accuracy = accuracy_score(all_labels, all_preds)  # Accuracy score
precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)  # Precision score
f1 = f1_score(all_labels, all_preds, average='macro')  # F1 score

# Display evaluation results
print("\nEvaluation Results:")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-Score: {f1:.4f}")
