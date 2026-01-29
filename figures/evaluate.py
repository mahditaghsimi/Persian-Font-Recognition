# evaluate.py - Model Evaluation and Analysis

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
    # Current file path
    current_dir = Path(__file__).parent.absolute()
    project_root = current_dir.parent if current_dir.name == 'figures' else current_dir

    # Search for model
    model_path = None
    encoder_path = None

    # First search in model folder
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
        # fallback to data
        data_dir = project_root / 'data'
        if data_dir.exists():
            data_path = str(data_dir)

    return {
        'model_path': model_path,
        'encoder_path': encoder_path,
        'data_path': data_path,
        'output_dir': str(current_dir)  # Output in the same location as the file
    }


# ==========================
# 1. Model Architecture
# ==========================

class DeepFontNet(nn.Module):
    def __init__(self, num_classes):
        super(DeepFontNet, self).__init__()

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

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn6 = nn.BatchNorm2d(64)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(64, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = self.relu(self.bn4(self.conv4(x)))

        x = self.relu(self.bn5(self.deconv1(x)))
        x = self.relu(self.bn6(self.deconv2(x)))

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


# ==========================
# 2. Dataset Class
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
# 3. Data Loader
# ==========================

def load_test_data(data_dir, label_encoder):
    """
    Load all data and split into test set
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
                img = cv2.resize(img, (105, 105))
                img = img.astype('float32') / 255.0
                images.append(img)
                labels.append(font_name)

    images = np.array(images).reshape(-1, 1, 105, 105)
    labels = label_encoder.transform(labels)

    # Split like training
    _, X_test, _, y_test = train_test_split(
        images, labels,
        test_size=0.25,
        random_state=42,
        stratify=labels
    )

    print(f"  Test set: {len(X_test)} images\n")

    return X_test, y_test


# ==========================
# 4. Visualization Functions
# ==========================

def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    """
    Plot Confusion Matrix
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(max(12, len(class_names) * 0.8), max(10, len(class_names) * 0.7)))

    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create annotation
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
    """
    Plot accuracy chart for each class
    """
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)

    # Calculate accuracy for each class
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

    # 1. Accuracy per class
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
    """
    Plot ROC Curve for all classes
    """
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

    # Calculate ROC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((all_labels == i).astype(int), all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot
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
    """
    Calculate and plot Top-K Accuracy
    """
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

    # Plot
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

    # Add values on bars
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
    """
    Save Classification Report as text file
    """
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
    """
    Display sample predictions
    """
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
                images_list.append(images[i].cpu().numpy())
                labels_list.append(labels[i].item())
                preds_list.append(predicted[i].item())
                probs_list.append(probs[i].max().item())

            if len(images_list) >= num_samples:
                break

    # Plot
    rows = 4
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))

    for idx in range(min(num_samples, len(images_list))):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        img = images_list[idx].squeeze()
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
    """
    Prediction confidence distribution
    """
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

    # Plot
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

    # Add means
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
    """
    Error analysis: which fonts have the most errors
    """
    # Find fonts with the most errors
    per_class_errors = []

    for i in range(len(class_names)):
        mask = (y_true == i)
        if mask.sum() > 0:
            errors = (y_pred[mask] != y_true[mask]).sum()
            error_rate = errors / mask.sum() * 100
            per_class_errors.append((class_names[i], error_rate, errors, mask.sum()))

    # Sort by error rate
    per_class_errors.sort(key=lambda x: x[1], reverse=True)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Error Rate
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

    # 2. Number of Errors
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
# 5. Main Evaluation Function
# ==========================

def evaluate_model(model_path=None,
                   encoder_path=None,
                   data_dir=None,
                   output_dir=None,
                   batch_size=32,
                   k_values=[1, 3, 5, 10],
                   num_sample_predictions=20):
    """
    Complete model evaluation and generate all charts
    """

    print("\n" + "=" * 80)
    print("PERSIAN FONT RECOGNITION - MODEL EVALUATION")
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
    print("Loading trained model...")
    model = DeepFontNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded successfully")
    print(f"  Total parameters: {total_params:,}\n")

    # Load test data
    print("Loading test data...")
    X_test, y_test = load_test_data(data_dir, label_encoder)

    # Create DataLoader
    test_dataset = FontDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

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

    # 1. Confusion Matrix
    print("[1/8] Generating confusion matrix...")
    plot_confusion_matrix(all_labels, all_preds, label_encoder.classes_, save_dir=output_dir)

    # 2. Per-class metrics
    print("[2/8] Generating per-class metrics...")
    plot_per_class_metrics(all_labels, all_preds, label_encoder.classes_, save_dir=output_dir)

    # 3. ROC Curves
    print("[3/8] Generating ROC curves...")
    roc_auc_scores = plot_roc_curves(model, test_loader, label_encoder, device, save_dir=output_dir)

    # 4. Top-K Accuracy
    print("[4/8] Computing Top-K accuracy...")
    top_k_scores = plot_top_k_accuracy(model, test_loader, device, k_values=k_values, save_dir=output_dir)

    # 5. Classification Report
    print("[5/8] Saving classification report...")
    save_classification_report(all_labels, all_preds, label_encoder.classes_, save_dir=output_dir)

    # 6. Sample Predictions
    print("[6/8] Generating sample predictions...")
    plot_sample_predictions(model, test_loader, label_encoder, device,
                            num_samples=num_sample_predictions, save_dir=output_dir)

    # 7. Confidence Distribution
    print("[7/8] Analyzing confidence distribution...")
    plot_prediction_confidence_distribution(model, test_loader, device, save_dir=output_dir)

    # 8. Error Analysis
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
# 6. Run Evaluation
# ==========================

if __name__ == "__main__":
    # Run evaluation with automatic path detection
    results = evaluate_model()

    print("Evaluation completed successfully!")
    print("Check the output folder for all visualizations!")
