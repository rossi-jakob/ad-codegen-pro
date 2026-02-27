import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
import torch.nn as nn
import torch.optim as optim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4

MODEL_PATH = "./local_vit_model"

# Load local processor & model
processor = AutoImageProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForImageClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model.to(DEVICE)

# Dataset
transform = Compose([
    Resize((224, 224)),
    ToTensor()
])

train_dataset = datasets.ImageFolder("dataset/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

num_classes = len(train_dataset.classes)

# Replace classifier head
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(pixel_values=images).logits
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "offline_trained_model.pth")