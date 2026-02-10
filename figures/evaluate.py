# evaluate.py - Model Evaluation and Analysis (4-Channel Version with Edge & Corner Features)

import os
import sys
from pathlib import Path
import numpy as np
import cv2
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_fscore_support
)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import warnings

warnings.filterwarnings('ignore')

# Font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


# ==========================
# 0. Auto Path Detection
# ==========================

def find_project_files():
    """
    Automatically find model and data file paths
    """
    current_dir = Path(__file__).parent.absolute()
    project_root = current_dir.parent if current_dir.name == 'figures' else current_dir

    model_path = None
    encoder_path = None

    # Search in model folder
    model_dir = project_root / 'model'
    if model_dir.exists():
        model_file = model_dir / 'persian_font_model.pth'
        encoder_file = model_dir / 'font_names.pkl'
        if model_file.exists() and encoder_file.exists():
            model_path = str(model_file)
            encoder_path = str(encoder_file)

    # If not found, search in current folder
    if not model_path:
        model_file = current_dir / 'persian_font_model.pth'
        encoder_file = current_dir / 'font_names.pkl'
        if model_file.exists() and encoder_file.exists():
            model_path = str(model_file)
            encoder_path = str(encoder_file)

    # Search for data
    data_path = None
    data_aug_dir = project_root / 'data_augmentations'
    if data_aug_dir.exists():
        data_path = str(data_aug_dir)
    else:
        data_dir = project_root / 'data'
        if data_dir.exists():
            data_path = str(data_dir)

    return {
        'model_path': model_path,
        'encoder_path': encoder_path,
        'data_path': data_path,
        'output_dir': str(current_dir)
    }


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
    # Resize to target size
    if isinstance(image, np.ndarray):
        image = cv2.resize(image, (105, 105))
    else:
        image = image.resize((105, 105))
        image = np.array(image)

    # Extract features
    edge_features = extract_edge_features(image)
    corner_features = extract_corner_features(image)

    # Calculate gradient magnitude
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_mag = np.uint8(gradient_mag / gradient_mag.max() * 255)

    # Stack all channels: [original, edges, corners, gradient]
    combined = np.stack([image, edge_features, corner_features, gradient_mag], axis=0)

    # Normalize to [0, 1]
    combined = combined.astype(np.float32) / 255.0

    return combined


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


# ===================== Improved Model (4-Channel Input) =====================
class ImprovedDeepFontNet(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedDeepFontNet, self).__init__()

        # Multi-scale first block (accepts 4 input channels: original, edges, corners, gradient)
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


# ==========================
# Dataset Class
# ==========================

class FontDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# ==========================
# Data Loader with 4-Channel Features
# ==========================

def load_test_data(data_dir, label_encoder):
    """
    Load all data with 4-channel feature extraction and split into test set
    """
    from sklearn.model_selection import train_test_split

    data_dir = Path(data_dir)
    images = []
    labels = []

    print(f"Loading data from: {data_dir}")

    for font_dir in sorted(data_dir.iterdir()):
        if not font_dir.is_dir():
            continue

        font_name = font_dir.name

        for img_file in font_dir.glob('*.png'):
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Apply 4-channel feature extraction
                img_features = preprocess_image_with_features(img)
                images.append(img_features)
                labels.append(font_name)

    images = np.array(images)  # Shape: (N, 4, 105, 105)
    labels = label_encoder.transform(labels)

    # Split like training (80/20)
    _, X_test, _, y_test = train_test_split(
        images, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    print(f"  Test set: {len(X_test)} images\n")

    return X_test, y_test


# ==========================
# Visualization Functions
# ==========================

def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    """Plot Confusion Matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(max(12, len(class_names) * 0.8), max(10, len(class_names) * 0.7)))

    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'},
                linewidths=0.5, linecolor='gray')

    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix with Percentages', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '01_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 01_confusion_matrix.png")


def plot_per_class_metrics(y_true, y_pred, class_names, save_dir):
    """Plot accuracy chart for each class"""
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)

    per_class_acc = []
    for i in range(len(class_names)):
        mask = (y_true == i)
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_true[mask]).sum() / mask.sum() * 100
            per_class_acc.append(acc)
        else:
            per_class_acc.append(0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    x_pos = np.arange(len(class_names))

    # 1. Accuracy
    bars1 = axes[0, 0].bar(x_pos, per_class_acc, color='steelblue', alpha=0.8, edgecolor='black')
    axes[0, 0].set_xlabel('Font Name', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 105])
    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 2. Precision
    bars2 = axes[0, 1].bar(x_pos, precision * 100, color='green', alpha=0.8, edgecolor='black')
    axes[0, 1].set_xlabel('Font Name', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Per-Class Precision', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim([0, 105])
    for bar in bars2:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 3. Recall
    bars3 = axes[1, 0].bar(x_pos, recall * 100, color='orange', alpha=0.8, edgecolor='black')
    axes[1, 0].set_xlabel('Font Name', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Per-Class Recall', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_ylim([0, 105])
    for bar in bars3:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 4. F1-Score
    bars4 = axes[1, 1].bar(x_pos, f1 * 100, color='red', alpha=0.8, edgecolor='black')
    axes[1, 1].set_xlabel('Font Name', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Per-Class F1-Score', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim([0, 105])
    for bar in bars4:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '02_per_class_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 02_per_class_metrics.png")


def plot_roc_curves(model, test_loader, label_encoder, device, save_dir):
    """Plot ROC Curve for all classes"""
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    n_classes = len(label_encoder.classes_)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((all_labels == i).astype(int), all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(14, 10))

    colors = cycle(plt.cm.tab20.colors)

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{label_encoder.classes_[i]} (AUC = {roc_auc[i]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('ROC Curves for All Classes', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '03_roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 03_roc_curves.png")

    return roc_auc


def plot_top_k_accuracy(model, test_loader, device, k_values, save_dir):
    """Calculate and plot Top-K Accuracy"""
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    top_k_accs = []

    for k in k_values:
        top_k_preds = np.argsort(all_probs, axis=1)[:, -k:]
        correct = sum([label in preds for label, preds in zip(all_labels, top_k_preds)])
        acc = correct / len(all_labels) * 100
        top_k_accs.append(acc)

    plt.figure(figsize=(12, 7))

    colors = ['steelblue', 'green', 'orange', 'red']
    bars = plt.bar(range(len(k_values)), top_k_accs, color=colors[:len(k_values)],
                   alpha=0.8, edgecolor='black')

    plt.xlabel('Top-K', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title('Top-K Accuracy', fontsize=16, fontweight='bold')
    plt.xticks(range(len(k_values)), [f'Top-{k}' for k in k_values])
    plt.ylim([0, 105])
    plt.grid(True, alpha=0.3, axis='y')

    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{top_k_accs[i]:.2f}%', ha='center', va='bottom',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '04_top_k_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 04_top_k_accuracy.png")

    return dict(zip([f'top_{k}' for k in k_values], top_k_accs))


def save_classification_report(y_true, y_pred, class_names, save_dir):
    """Save Classification Report as text file"""
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    report_path = os.path.join(save_dir, '05_classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PERSIAN FONT RECOGNITION - CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
        f.write("\n" + "=" * 80 + "\n")

    print(f"  Saved: 05_classification_report.txt")


def plot_sample_predictions(model, test_loader, label_encoder, device, num_samples, save_dir):
    """Display sample predictions"""
    model.eval()

    images_list = []
    labels_list = []
    preds_list = []
    probs_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images_batch = images.to(device)
            outputs = model(images_batch)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)

            for i in range(min(len(images), num_samples - len(images_list))):
                # Use only the first channel (original image) for display
                images_list.append(images[i][0].cpu().numpy())
                labels_list.append(labels[i].item())
                preds_list.append(predicted[i].item())
                probs_list.append(probs[i].max().item())

            if len(images_list) >= num_samples:
                break

    rows = 4
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))

    for idx in range(min(num_samples, len(images_list))):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        img = images_list[idx]
        true_label = label_encoder.classes_[labels_list[idx]]
        pred_label = label_encoder.classes_[preds_list[idx]]
        confidence = probs_list[idx] * 100

        ax.imshow(img, cmap='gray')
        ax.axis('off')

        color = 'green' if labels_list[idx] == preds_list[idx] else 'red'
        title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%'
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')

    plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '06_sample_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 06_sample_predictions.png")


def plot_prediction_confidence_distribution(model, test_loader, device, save_dir):
    """Prediction confidence distribution"""
    model.eval()

    correct_confidences = []
    wrong_confidences = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probs, 1)

            correct_mask = (predicted == labels)

            correct_confidences.extend(confidences[correct_mask].cpu().numpy())
            wrong_confidences.extend(confidences[~correct_mask].cpu().numpy())

    plt.figure(figsize=(12, 7))

    plt.hist(correct_confidences, bins=50, alpha=0.7, color='green',
             label=f'Correct Predictions (n={len(correct_confidences)})', edgecolor='black')
    plt.hist(wrong_confidences, bins=50, alpha=0.7, color='red',
             label=f'Wrong Predictions (n={len(wrong_confidences)})', edgecolor='black')

    plt.xlabel('Prediction Confidence', fontsize=14, fontweight='bold')
    plt.ylabel('Frequency', fontsize=14, fontweight='bold')
    plt.title('Prediction Confidence Distribution', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    if len(correct_confidences) > 0:
        plt.axvline(np.mean(correct_confidences), color='darkgreen', linestyle='--',
                    linewidth=2, label=f'Correct Mean: {np.mean(correct_confidences):.3f}')
    if len(wrong_confidences) > 0:
        plt.axvline(np.mean(wrong_confidences), color='darkred', linestyle='--',
                    linewidth=2, label=f'Wrong Mean: {np.mean(wrong_confidences):.3f}')

    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '07_confidence_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 07_confidence_distribution.png")


def plot_error_analysis(y_true, y_pred, class_names, save_dir):
    """Error analysis: which fonts have the most errors"""
    per_class_errors = []

    for i in range(len(class_names)):
        mask = (y_true == i)
        if mask.sum() > 0:
            errors = (y_pred[mask] != y_true[mask]).sum()
            error_rate = errors / mask.sum() * 100
            per_class_errors.append((class_names[i], error_rate, errors, mask.sum()))

    per_class_errors.sort(key=lambda x: x[1], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    font_names = [x[0] for x in per_class_errors]
    error_rates = [x[1] for x in per_class_errors]

    colors = ['red' if rate > 20 else 'orange' if rate > 10 else 'green'
              for rate in error_rates]

    bars1 = axes[0].barh(range(len(font_names)), error_rates, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_xlabel('Error Rate (%)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Font Name', fontsize=14, fontweight='bold')
    axes[0].set_title('Per-Class Error Rate', fontsize=16, fontweight='bold')
    axes[0].set_yticks(range(len(font_names)))
    axes[0].set_yticklabels(font_names)
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')

    for i, bar in enumerate(bars1):
        width = bar.get_width()
        axes[0].text(width, bar.get_y() + bar.get_height() / 2.,
                     f'{error_rates[i]:.1f}%',
                     ha='left', va='center', fontsize=9, fontweight='bold')

    error_counts = [x[2] for x in per_class_errors]

    bars2 = axes[1].barh(range(len(font_names)), error_counts, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Number of Errors', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Font Name', fontsize=14, fontweight='bold')
    axes[1].set_title('Number of Errors per Class', fontsize=16, fontweight='bold')
    axes[1].set_yticks(range(len(font_names)))
    axes[1].set_yticklabels(font_names)
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')

    for i, bar in enumerate(bars2):
        width = bar.get_width()
        axes[1].text(width, bar.get_y() + bar.get_height() / 2.,
                     f'{int(error_counts[i])}',
                     ha='left', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '08_error_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 08_error_analysis.png")


# ==========================
# Main Evaluation Function
# ==========================

def evaluate_model(model_path=None,
                   encoder_path=None,
                   data_dir=None,
                   output_dir=None,
                   batch_size=32,
                   k_values=[1, 3, 5, 10],
                   num_sample_predictions=20):
    """Complete model evaluation and generate all charts"""

    print("\n" + "=" * 80)
    print("PERSIAN FONT RECOGNITION - MODEL EVALUATION (4-Channel with Edge & Corner)")
    print("=" * 80 + "\n")

    # Auto-detect paths if not provided
    if not all([model_path, encoder_path, data_dir, output_dir]):
        print("Auto-detecting project files...")
        paths = find_project_files()

        model_path = model_path or paths['model_path']
        encoder_path = encoder_path or paths['encoder_path']
        data_dir = data_dir or paths['data_path']
        output_dir = output_dir or paths['output_dir']

        print(f"  Model: {model_path}")
        print(f"  Encoder: {encoder_path}")
        print(f"  Data: {data_dir}")
        print(f"  Output: {output_dir}\n")

    # Validate paths
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not encoder_path or not Path(encoder_path).exists():
        raise FileNotFoundError(f"Encoder file not found: {encoder_path}")
    if not data_dir or not Path(data_dir).exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print("GPU not available, using CPU\n")

    # Load label encoder
    print("Loading label encoder...")
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    num_classes = len(label_encoder.classes_)
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {', '.join(label_encoder.classes_)}\n")

    # Load model
    print("Loading trained ImprovedDeepFontNet model (4-channel)...")
    model = ImprovedDeepFontNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded successfully")
    print(f"  Total parameters: {total_params:,}\n")

    # Load test data
    print("Loading test data with 4-channel feature extraction...")
    X_test, y_test = load_test_data(data_dir, label_encoder)

    # Create DataLoader
    test_dataset = FontDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda
    )

    print(f"Test batches: {len(test_loader)}\n")

    # Get predictions
    print("Generating predictions on test set...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate overall accuracy
    overall_accuracy = (all_preds == all_labels).mean() * 100
    print(f"  Overall Accuracy: {overall_accuracy:.2f}%\n")

    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80 + "\n")

    # Generate all visualizations
    print("[1/8] Generating confusion matrix...")
    plot_confusion_matrix(all_labels, all_preds, label_encoder.classes_, save_dir=output_dir)

    print("[2/8] Generating per-class metrics...")
    plot_per_class_metrics(all_labels, all_preds, label_encoder.classes_, save_dir=output_dir)

    print("[3/8] Generating ROC curves...")
    roc_auc_scores = plot_roc_curves(model, test_loader, label_encoder, device, save_dir=output_dir)

    print("[4/8] Computing Top-K accuracy...")
    top_k_scores = plot_top_k_accuracy(model, test_loader, device, k_values=k_values, save_dir=output_dir)

    print("[5/8] Saving classification report...")
    save_classification_report(all_labels, all_preds, label_encoder.classes_, save_dir=output_dir)

    print("[6/8] Generating sample predictions...")
    plot_sample_predictions(model, test_loader, label_encoder, device,
                            num_samples=num_sample_predictions, save_dir=output_dir)

    print("[7/8] Analyzing confidence distribution...")
    plot_prediction_confidence_distribution(model, test_loader, device, save_dir=output_dir)

    print("[8/8] Performing error analysis...")
    plot_error_analysis(all_labels, all_preds, label_encoder.classes_, save_dir=output_dir)

    # Final Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"  Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"  Top-3 Accuracy: {top_k_scores['top_3']:.2f}%")
    print(f"  Top-5 Accuracy: {top_k_scores['top_5']:.2f}%")
    if 'top_10' in top_k_scores:
        print(f"  Top-10 Accuracy: {top_k_scores['top_10']:.2f}%")
    print(f"  Average AUC: {np.mean(list(roc_auc_scores.values())):.4f}")
    print(f"  Total samples evaluated: {len(all_labels)}")
    print(f"  Correct predictions: {(all_preds == all_labels).sum()}")
    print(f"  Wrong predictions: {(all_preds != all_labels).sum()}")
    print("=" * 80 + "\n")

    print("=" * 80)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print(f"All files saved to: {os.path.abspath(output_dir)}")
    print("=" * 80 + "\n")

    return {
        'overall_accuracy': overall_accuracy,
        'top_k_scores': top_k_scores,
        'roc_auc_scores': roc_auc_scores,
        'predictions': all_preds,
        'true_labels': all_labels
    }


# ==========================
# Run Evaluation
# ==========================

if __name__ == "__main__":
    results = evaluate_model()
    print("Evaluation completed successfully!")
    print("Check the output folder for all visualizations!")
