import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# ======================
# Configuration
# ======================
DATA_DIR = "dataset"
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4

# ======================
# Data Transforms
# ======================
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ======================
# Load Dataset
# ======================
train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"),
    transform=val_transforms
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)

# ======================
# Load Pretrained Model
# ======================
model = models.resnet50(pretrained=True)

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

# ======================
# Loss & Optimizer
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# ======================
# Training Loop
# ======================
best_val_acc = 0.0
patience = 5
patience_counter = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # ---- Training ----
    model.train()
    train_loss = 0
    train_correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = train_correct / total
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    # ---- Validation ----
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    print(f"Val Acc: {val_acc:.4f}")

    # ---- Save Best Model ----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Best model saved")
        patience_counter = 0
    else:
        patience_counter += 1

    # ---- Early Stopping ----
    if patience_counter >= patience:
        print("⛔ Early stopping triggered")
        break

print("Training Complete.")
print("Best Validation Accuracy:", best_val_acc)