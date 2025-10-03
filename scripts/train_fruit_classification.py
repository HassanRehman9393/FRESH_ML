#!/usr/bin/env python3
"""
Fruit Classification Training Script
Train a CNN model for 16-class fruit classification including ripeness levels and varieties
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
from pathlib import Path
import yaml
import time
from datetime import datetime

def load_dataset_config():
    """Load dataset configuration"""
    config_path = Path('data/unified/fruit_classification/dataset.yaml')
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_data_transforms():
    """Define data augmentation and preprocessing transforms"""
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_test_transforms

def create_model(num_classes=16, model_type='resnet50'):
    """Create and configure the CNN model"""
    if model_type == 'resnet50':
        model = models.resnet50(pretrained=True)
        # Freeze early layers
        for param in list(model.parameters())[:-10]:
            param.requires_grad = False
        # Replace final layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_type == 'efficientnet':
        model = models.efficientnet_b0(pretrained=True)
        # Freeze early layers
        for param in list(model.parameters())[:-10]:
            param.requires_grad = False
        # Replace classifier
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

def evaluate_model(model, test_loader, device, class_names):
    """Comprehensive model evaluation"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate accuracy
    accuracy = 100. * sum([all_preds[i] == all_targets[i] for i in range(len(all_preds))]) / len(all_preds)
    return accuracy

def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Training Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training function"""
    print("=" * 60)
    print("FRUIT CLASSIFICATION TRAINING")
    print("=" * 60)
    
    # Load dataset configuration
    try:
        config = load_dataset_config()
        class_names = config['names']
        num_classes = config['nc']
        print(f"Dataset: {num_classes} classes")
        print(f"Classes: {class_names}")
    except Exception as e:
        print(f"Error loading dataset config: {e}")
        return
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    MODEL_TYPE = 'resnet50'  # or 'efficientnet'
    
    print(f"\nTraining Parameters:")
    print(f"Model: {MODEL_TYPE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Data paths
    data_dir = Path('data/unified/fruit_classification')
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'
    
    # Check if directories exist
    for dir_path in [train_dir, val_dir, test_dir]:
        if not dir_path.exists():
            print(f"Error: Directory not found: {dir_path}")
            return
    
    # Get transforms
    train_transforms, val_test_transforms = get_data_transforms()
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Create model
    print(f"\nCreating {MODEL_TYPE} model...")
    model = create_model(num_classes, MODEL_TYPE)
    model = model.to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    # Training history
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    
    print("\nStarting training...")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        print('-' * 30)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_fruit_classification_model.pth')
            print(f'New best model saved! Accuracy: {val_acc:.2f}%')
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('best_fruit_classification_model.pth'))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy = evaluate_model(model, test_loader, device, class_names)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Plot training history
    plot_training_history(train_losses, train_accs, val_losses, val_accs)
    
    # Save final model and training info
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': MODEL_TYPE,
        'num_classes': num_classes,
        'class_names': class_names,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE
    }, 'fruit_classification_final.pth')
    
    print(f"\nTraining Summary:")
    print(f"Model saved as: fruit_classification_final.pth")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print(f"Training time: {training_time/3600:.2f} hours")
    print("Training completed successfully!")

if __name__ == "__main__":
    main()