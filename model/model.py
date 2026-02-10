import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder
import pickle
import cv2
import numpy as np
from torch.cuda.amp import autocast, GradScaler


# ===================== Feature Extraction Functions =====================
def extract_edge_features(image):
    """Extract edge features using Canny and Sobel operators"""
    img_np = np.array(image)

    # Canny edge detection
    edges_canny = cv2.Canny(img_np, 50, 150)

    # Sobel edge detection
    sobel_x = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_combined = np.uint8(sobel_combined / sobel_combined.max() * 255)

    # Combine features
    edge_features = np.maximum(edges_canny, sobel_combined)

    return edge_features


def extract_corner_features(image):
    """Extract corner features using Harris Corner Detection"""
    img_np = np.array(image)

    # Harris corner detection
    img_float = np.float32(img_np)
    corners = cv2.cornerHarris(img_float, blockSize=2, ksize=3, k=0.04)
    corners = cv2.dilate(corners, None)

    # Normalize
    corners_normalized = np.uint8(corners / corners.max() * 255)

    return corners_normalized


def preprocess_image_with_features(image):
    """
    Preprocess image and extract structural features
    Returns a 4-channel tensor: [original, edges, corners, gradient]
    """
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')

    # Resize to target size
    image = image.resize((105, 105))

    # Extract features
    edge_features = extract_edge_features(image)
    corner_features = extract_corner_features(image)

    # Convert original image to numpy
    img_np = np.array(image)

    # Calculate gradient magnitude
    gradient_x = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_mag = np.uint8(gradient_mag / gradient_mag.max() * 255)

    # Stack all channels
    combined = np.stack([img_np, edge_features, corner_features, gradient_mag], axis=0)

    # Normalize to [0, 1]
    combined = combined.astype(np.float32) / 255.0

    # Convert to tensor
    tensor = torch.from_numpy(combined)

    return tensor


# ===================== Channel and Spatial Attention =====================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


# ===================== Improved Model =====================
class ImprovedDeepFontNet(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedDeepFontNet, self).__init__()

        # Multi-scale first block (accepts 4 input channels)
        self.conv1a = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(4, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.ca1 = ChannelAttention(128)
        self.sa1 = SpatialAttention()

        # Second block
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.ca2 = ChannelAttention(256)
        self.sa2 = SpatialAttention()

        # Third block
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.ca3 = ChannelAttention(512)
        self.sa3 = SpatialAttention()

        # Global pooling (both avg and max)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 2, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, num_classes)

        self.leaky_relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Multi-scale first block
        x1a = self.relu(self.conv1a(x))
        x1b = self.relu(self.conv1b(x))
        x = torch.cat([x1a, x1b], dim=1)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.ca1(x)
        x = self.sa1(x)

        # Second block
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.ca2(x)
        x = self.sa2(x)

        # Third block
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.ca3(x)
        x = self.sa3(x)

        # Global pooling
        avg_pool = self.global_avg_pool(x).view(x.size(0), -1)
        max_pool = self.global_max_pool(x).view(x.size(0), -1)
        x = torch.cat([avg_pool, max_pool], dim=1)

        # FC layers
        x = self.leaky_relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)

        x = self.leaky_relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.fc3(x)

        return x


# ===================== Label Smoothing Loss =====================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = torch.nn.functional.log_softmax(pred, dim=-1)

        loss = -log_preds.sum(dim=-1).mean()
        nll = torch.nn.functional.nll_loss(log_preds, target, reduction='mean')

        return (1 - self.smoothing) * nll + self.smoothing * (loss / n_classes)


# ===================== Dataset Class =====================
class FontDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = []
        self.labels = []

        for font_name in os.listdir(data_dir):
            font_dir = os.path.join(data_dir, font_name)
            if os.path.isdir(font_dir):
                for img_name in os.listdir(font_dir):
                    img_path = os.path.join(font_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(font_name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('L')

        # Apply feature extraction
        image_tensor = preprocess_image_with_features(image)

        return image_tensor, label


# ===================== Training Function =====================
def train_model(data_dir='../data_augmentations', batch_size=32, num_epochs=50, learning_rate=0.001, save_dir='./'):
    """
    Train the improved model with feature engineering
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()
    print(f"Using device: {device}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = FontDataset(data_dir)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create data loaders with conditional pin_memory
    print(f"\nCreating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda  # Only True if GPU available
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda  # Only True if GPU available
    )

    # Encode labels
    print("\nEncoding labels...")
    label_encoder = LabelEncoder()
    all_labels = [label for _, label in dataset]
    label_encoder.fit(all_labels)
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")

    # Initialize model
    print("\nInitializing ImprovedDeepFontNet model...")
    model = ImprovedDeepFontNet(num_classes).to(device)

    # Loss and optimizer
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    # Mixed precision training
    scaler = GradScaler() if use_cuda else None

    # Training loop
    print("\nStarting training...")
    best_accuracy = 0.0
    patience = 15
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels_encoded = torch.tensor(label_encoder.transform(labels)).to(device)

            optimizer.zero_grad()

            if use_cuda and scaler is not None:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels_encoded)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels_encoded)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels_encoded = torch.tensor(label_encoder.transform(labels)).to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels_encoded.size(0)
                correct += (predicted == labels_encoded).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Accuracy: {accuracy:.2f}%')

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'persian_font_model.pth'))
            with open(os.path.join(save_dir, 'font_names.pkl'), 'wb') as f:
                pickle.dump(label_encoder, f)
            print(f'Model saved with accuracy: {best_accuracy:.2f}%')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    print(f'\nTraining completed. Best accuracy: {best_accuracy:.2f}%')
    return model, label_encoder


# ===================== Prediction Function =====================
def predict_font(image_path, model_path='persian_font_model.pth', encoder_path='font_names.pkl'):
    """
    Predict font from an image using the improved model with features
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load label encoder
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    num_classes = len(label_encoder.classes_)

    # Load model
    model = ImprovedDeepFontNet(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert('L')
    image_tensor = preprocess_image_with_features(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Get top 3 predictions
        top3_prob, top3_idx = torch.topk(probabilities, 3)

        print("\nTop 3 Predictions:")
        for i in range(3):
            font_name = label_encoder.inverse_transform([top3_idx[0][i].item()])[0]
            confidence = top3_prob[0][i].item() * 100
            print(f"{i + 1}. {font_name}: {confidence:.2f}%")

        # Return best prediction
        predicted_class = top3_idx[0][0].item()
        predicted_font = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_font


# ===================== Main Execution =====================
if __name__ == "__main__":
    # Train the model
    model, label_encoder = train_model(
        data_dir='../data_augmentations',
        batch_size=32,
        num_epochs=50,
        learning_rate=0.001,
        save_dir='./'
    )

    # Example prediction
    # predicted_font = predict_font('path_to_test_image.png')
    # print(f'\nPredicted font: {predicted_font}')