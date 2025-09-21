import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
import numpy as np
import os
from PIL import Image
import random
import copy
from pathlib import Path
from tqdm import tqdm


class StanfordDogsDataset(Dataset):
    """Custom dataset for Stanford Dogs dataset"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        # Get all breed directories
        image_dir = self.data_dir / "images" / "Images"
        breed_dirs = sorted([d for d in image_dir.iterdir() if d.is_dir()])
        
        # Create class mapping
        for idx, breed_dir in enumerate(breed_dirs):
            breed_name = breed_dir.name.split('-', 1)[1]  # Remove prefix
            self.class_to_idx[breed_name] = idx
        
        # Load all images and labels
        for breed_idx, breed_dir in enumerate(breed_dirs):
            for img_path in breed_dir.glob("*.jpg"):
                self.images.append(str(img_path))
                self.labels.append(breed_idx)
        
        self.num_classes = len(breed_dirs)
        print(f"Loaded {len(self.images)} images from {self.num_classes} classes")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image in case of error
            dummy_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, label


class MeanTeacherModel:
    """Mean Teacher implementation with ResNet backbone"""
    
    def __init__(self, num_classes, ema_decay=0.999):
        self.num_classes = num_classes
        self.ema_decay = ema_decay
        
        # Create student and teacher models
        weights = ResNet18_Weights.DEFAULT
        self.student = models.resnet18(weights)
        self.student.fc = nn.Linear(self.student.fc.in_features, num_classes)

        self.teacher = models.resnet18(weights)
        self.teacher.fc = nn.Linear(self.teacher.fc.in_features, num_classes)
        
        # Initialize teacher with student weights
        self.teacher.load_state_dict(self.student.state_dict())
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def update_teacher(self):
        """Update teacher model using exponential moving average"""
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
                teacher_param.data = (self.ema_decay * teacher_param.data + 
                                    (1.0 - self.ema_decay) * student_param.data)
    
    def to(self, device):
        self.student = self.student.to(device)
        self.teacher = self.teacher.to(device)
        return self


def get_transforms():
    """Get weak and strong augmentation transforms"""
    # Weak augmentation (for teacher)
    weak_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Strong augmentation (for student)
    strong_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return weak_transform, strong_transform, test_transform


def consistency_loss(student_logits, teacher_logits, temperature=1.0):
    """Compute consistency loss between student and teacher predictions"""
    student_probs = F.softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.mse_loss(student_probs, teacher_probs)


def train_teacher_model(model, train_loader, optimizer, device, consistency_weight=1.0):
    """Train for one epoch using Mean Teacher"""
    model.student.train()
    model.teacher.eval()
    
    total_loss = 0
    total_class_loss = 0
    total_consistency_loss = 0
    correct = 0
    total = 0
    
    # Add progress bar for batches    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Apply different augmentations to the same batch
        # For simplicity, we'll use the same images but could apply different augs
        weak_images = images  # Assume images already have weak augmentation
        strong_images = images  # In practice, apply different augmentation
        
        optimizer.zero_grad()
        
        # Student predictions on both weak and strong augmentations
        student_weak = model.student(weak_images)
        student_strong = model.student(strong_images)
        
        # Teacher predictions on weak augmentations only
        with torch.no_grad():
            teacher_weak = model.teacher(weak_images)
        
        # Classification loss (supervised)
        class_loss = F.cross_entropy(student_weak, labels)
        
        # Consistency loss (unsupervised)
        consistency_loss_val = consistency_loss(student_strong, teacher_weak)
        
        # Combined loss
        total_loss_val = class_loss + consistency_weight * consistency_loss_val
        
        total_loss_val.backward()
        optimizer.step()
        
        # Update teacher using EMA
        model.update_teacher()
        
        # Statistics
        total_loss += total_loss_val.item()
        total_class_loss += class_loss.item()
        total_consistency_loss += consistency_loss_val.item()
        
        _, predicted = student_weak.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{total_loss_val.item():.4f}',
            'Class': f'{class_loss.item():.4f}',
            'Consistency': f'{consistency_loss_val.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss/len(train_loader), total_class_loss/len(train_loader), total_consistency_loss/len(train_loader), 100.*correct/total


def evaluate(model, test_loader, device):
    """Evaluate the model"""
    model.teacher.eval()
    correct = 0
    total = 0
    
    # Add progress bar for evaluation
    pbar = tqdm(test_loader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model.teacher(images)  # Use teacher for evaluation
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    accuracy = 100. * correct / total
    return accuracy


if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 20
    consistency_weight = 1.0
    ema_decay = 0.999
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data preparation
    data_dir = "data/stanford_dogs"
    weak_transform, strong_transform, test_transform = get_transforms()
    
    # Load dataset with weak transform (we'll handle strong augmentation in training loop)
    full_dataset = StanfordDogsDataset(data_dir, transform=weak_transform)
    
    # Split dataset (80% train, 20% test)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create Mean Teacher model
    model = MeanTeacherModel(num_classes=full_dataset.num_classes, ema_decay=ema_decay)
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.student.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    print("Starting training...")
    
    # Training loop with progress bar for epochs
    best_acc = 0
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress")
    
    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, class_loss, consistency_loss_val, train_acc = train_teacher_model(
            model, train_loader, optimizer, device, consistency_weight
        )
        
        # Evaluate
        test_acc = evaluate(model, test_loader, device)
        
        # Update learning rate
        scheduler.step()
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'Train Acc': f'{train_acc:.2f}%',
            'Test Acc': f'{test_acc:.2f}%',
            'Best Acc': f'{best_acc:.2f}%'
        })
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Results:")
        print(f"Train Loss: {train_loss:.4f}, Class Loss: {class_loss:.4f}, "
              f"Consistency Loss: {consistency_loss_val:.4f}")
        print(f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'student_state_dict': model.student.state_dict(),
                'teacher_state_dict': model.teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, 'best_mean_teacher_model.pth')
            print(f"✓ New best model saved! Accuracy: {best_acc:.2f}%")
    
    print(f"\n🎉 Training completed! Best test accuracy: {best_acc:.2f}%")