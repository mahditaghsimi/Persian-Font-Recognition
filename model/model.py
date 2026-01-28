import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import gc
from pathlib import Path
import time


# ==========================
# 1. DeepFont Model (PyTorch)
# ==========================

class DeepFontNet(nn.Module):
    def __init__(self, num_classes):
        super(DeepFontNet, self).__init__()

        # Encoder (Convolutional layers)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Decoder (Transpose Convolution)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn6 = nn.BatchNorm2d(64)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Dense layers
        self.fc1 = nn.Linear(64, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Encoder
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = self.relu(self.bn4(self.conv4(x)))

        # Decoder
        x = self.relu(self.bn5(self.deconv1(x)))
        x = self.relu(self.bn6(self.deconv2(x)))

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Dense layers
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


# ==========================
# 2. Custom Dataset Class
# ==========================

class FontDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)
        print(f"  Dataset initialized: {len(self.images)} samples")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# ==========================
# 3. Data Loader
# ==========================

class DatasetLoader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        print(f"Data directory: {self.data_dir.absolute()}")

    def load_data(self):
        images = []
        labels = []
        font_names = []

        print("\nLoading dataset...")
        print("=" * 60)

        for font_dir in sorted(self.data_dir.iterdir()):
            if not font_dir.is_dir():
                continue

            font_name = font_dir.name
            font_names.append(font_name)

            img_count = 0
            for img_file in font_dir.glob('*.png'):
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (105, 105))
                    img = img.astype('float32') / 255.0
                    images.append(img)
                    labels.append(font_name)
                    img_count += 1

            print(f"  {font_name:30s}: {img_count:3d} images")

        print("=" * 60)
        print(f"Total: {len(images)} images from {len(font_names)} fonts\n")

        return np.array(images), np.array(labels), font_names


# ==========================
# 4. Training Function
# ==========================

def train_model(data_dir=None,
                batch_size=16,
                epochs=50,
                test_size=0.25,
                learning_rate=0.001,
                save_dir='./'):
    if data_dir is None:
        project_root = Path(__file__).parent.parent.absolute()
        data_dir = project_root / 'data_augmentations'

    print("\n" + "=" * 60)
    print("PERSIAN FONT CLASSIFICATION TRAINING")
    print("=" * 60 + "\n")

    # Detect device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        torch.cuda.empty_cache()
    else:
        print("GPU not available, using CPU")

    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Test size: {test_size * 100}%\n")

    # Memory cleanup
    gc.collect()

    # Load data
    start_time = time.time()
    loader = DatasetLoader(data_dir)
    images, labels, font_names = loader.load_data()
    print(f"Data loading time: {time.time() - start_time:.2f}s\n")

    # Reshape for PyTorch: (N, C, H, W)
    print("Reshaping images for PyTorch...")
    images = images.reshape(-1, 1, 105, 105)
    print(f"  Image shape: {images.shape}\n")

    gc.collect()

    # Encode labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)

    print("=" * 60)
    print(f"Number of classes: {num_classes}")
    print(f"Class names:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  {i:2d}. {class_name}")
    print("=" * 60 + "\n")

    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels_encoded,
        test_size=test_size,
        random_state=42,
        stratify=labels_encoded
    )

    print(f"  Training set: {len(X_train):5d} images ({len(X_train) / len(images) * 100:.1f}%)")
    print(f"  Test set:     {len(X_test):5d} images ({len(X_test) / len(images) * 100:.1f}%)\n")

    # Memory cleanup
    del images, labels, labels_encoded
    gc.collect()

    # Create Dataset and DataLoader
    print("Creating datasets...")
    train_dataset = FontDataset(X_train, y_train)
    test_dataset = FontDataset(X_test, y_test)

    print(f"\nCreating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}\n")

    # Build model
    print("Building model...")
    model = DeepFontNet(num_classes=num_classes).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print(f"Model Architecture:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 ** 2:.2f} MB")
    print("=" * 60 + "\n")

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )

    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0

    print("Starting training...")
    print("=" * 60 + "\n")

    training_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (batch_images, batch_labels) in enumerate(train_loader):
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_images, batch_labels in test_loader:
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

        val_loss /= len(test_loader)
        val_accuracy = 100 * val_correct / val_total

        # Learning rate scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start_time

        # Print results
        print(f"Epoch [{epoch + 1:2d}/{epochs}] | Time: {epoch_time:.1f}s")
        print(f"  Train | Loss: {train_loss:.4f} | Acc: {train_accuracy:6.2f}%")
        print(f"  Val   | Loss: {val_loss:.4f} | Acc: {val_accuracy:6.2f}%")

        # Show Learning Rate change
        if old_lr != new_lr:
            print(f"  Learning Rate: {old_lr:.2e} -> {new_lr:.2e}")

        # Early stopping and save best model
        is_best = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_accuracy
            patience_counter = 0
            is_best = True
            model_path = os.path.join(save_dir, 'persian_font_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"  Model saved! (Best Val Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"  Best validation accuracy: {best_val_acc:.2f}%")
                break

        print()

    training_time = time.time() - training_start_time

    print("=" * 60)
    print("Training completed!")
    print(f"Total training time: {training_time / 60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 60 + "\n")

    # Load best model
    print("Loading best model...")
    model_path = os.path.join(save_dir, 'persian_font_model.pth')
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # Save label encoder
    print("Saving label encoder...")
    encoder_path = os.path.join(save_dir, 'font_names.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)

    print("\n" + "=" * 60)
    print("Files saved:")
    print(f"  {model_path}")
    print(f"  {encoder_path}")
    print("=" * 60 + "\n")

    return model, label_encoder


# ==========================
# 5. Inference Function
# ==========================

def predict_font(model, image_path, label_encoder, device='cpu'):
    """
    Predict font for an image
    """
    print(f"\nPredicting font for: {image_path}")

    model.eval()

    # Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    original_shape = img.shape
    img = cv2.resize(img, (105, 105))
    img = img.astype('float32') / 255.0
    img = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = label_encoder.inverse_transform([predicted.item()])[0]

    print(f"  Predicted font: {predicted_class}")
    print(f"  Confidence: {confidence.item() * 100:.2f}%")

    return predicted_class, confidence.item()


# ==========================
# 6. Run Training
# ==========================

if __name__ == "__main__":
    model, label_encoder = train_model(
        data_dir='../data_augmentations',
        batch_size=32,
        epochs=50,
        test_size=0.25,
        learning_rate=0.001,
        save_dir='./'
    )

    print("Training pipeline completed successfully!")

    # Example usage for prediction
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # predicted_font, confidence = predict_font(model, 'test_image.png', label_encoder, device)
