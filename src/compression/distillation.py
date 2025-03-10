#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Knowledge distillation module for training smaller student models.
"""

import os
import logging
import time
from typing import Dict, Tuple, Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.baseline import load_baseline_model
from src.compression.quantization import get_model_size

logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """
    Distillation loss combining hard targets (ground truth) and soft targets (teacher predictions).
    """
    def __init__(self, alpha: float = 0.5, temperature: float = 2.0):
        """
        Args:
            alpha (float): Weight for the distillation loss (0.0 to 1.0)
                          alpha=0.0 means only use the hard target loss
                          alpha=1.0 means only use the soft target (distillation) loss
            temperature (float): Temperature for the soft targets
        """
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self, 
        student_outputs: torch.Tensor, 
        targets: torch.Tensor, 
        teacher_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the distillation loss.
        
        Args:
            student_outputs (torch.Tensor): Logits from the student model
            targets (torch.Tensor): Hard labels (ground truth)
            teacher_outputs (torch.Tensor): Logits from the teacher model
            
        Returns:
            torch.Tensor: Combined loss
        """
        # Hard targets loss (standard cross-entropy)
        hard_loss = self.ce_loss(student_outputs, targets)
        
        # Soft targets loss (KL divergence)
        soft_loss = F.kl_div(
            F.log_softmax(student_outputs / self.temperature, dim=1),
            F.softmax(teacher_outputs / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Combine the two losses
        return (1 - self.alpha) * hard_loss + self.alpha * soft_loss


def train_student_model(
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    alpha: float = 0.5,
    temperature: float = 2.0,
    epochs: int = 100,
    learning_rate: float = 0.01
) -> nn.Module:
    """
    Train a student model using knowledge distillation.
    
    Args:
        teacher_model (nn.Module): Pretrained teacher model
        student_model (nn.Module): Student model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        alpha (float): Weight for the distillation loss
        temperature (float): Temperature for the soft targets
        epochs (int): Number of training epochs
        learning_rate (float): Initial learning rate
        
    Returns:
        nn.Module: Trained student model
    """
    logger.info(f"Training student model with knowledge distillation for {epochs} epochs")
    logger.info(f"Distillation parameters: alpha={alpha}, temperature={temperature}")
    
    device = next(teacher_model.parameters()).device
    student_model = student_model.to(device)
    
    # Set up optimizer, scheduler, and loss function
    optimizer = torch.optim.SGD(
        student_model.parameters(), 
        lr=learning_rate, 
        momentum=0.9, 
        weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = DistillationLoss(alpha=alpha, temperature=temperature)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Best model tracking
    best_val_acc = 0.0
    temp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                            'results', 'temp_student_model.pth')
    
    # Teacher model should be in eval mode
    teacher_model.eval()
    
    for epoch in range(epochs):
        # Training phase
        student_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Use tqdm for progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass through both models
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
            
            student_outputs = student_model(inputs)
            
            # Compute loss
            loss = criterion(student_outputs, targets, teacher_outputs)
            
            # Backward + optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = student_outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': train_loss / train_total,
                'acc': 100. * train_correct / train_total
            })
        
        # Update learning rate
        scheduler.step()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        student_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]")
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass through both models
                teacher_outputs = teacher_model(inputs)
                student_outputs = student_model(inputs)
                
                # Compute loss
                loss = criterion(student_outputs, targets, teacher_outputs)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = student_outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': val_loss / val_total,
                    'acc': 100. * val_correct / val_total
                })
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100. * val_correct / val_total
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            logger.info(f"Validation accuracy improved from {best_val_acc:.2f}% to {val_acc:.2f}%")
            best_val_acc = val_acc
            torch.save(student_model.state_dict(), temp_path)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                   f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load the best model
    student_model.load_state_dict(torch.load(temp_path))
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return student_model


def compare_models(
    teacher_model: nn.Module,
    student_model: nn.Module,
    test_loader: DataLoader
) -> Dict[str, Dict[str, float]]:
    """
    Compare the performance of teacher and student models.
    
    Args:
        teacher_model (nn.Module): Teacher model
        student_model (nn.Module): Student model
        test_loader (DataLoader): Test data loader
        
    Returns:
        dict: Comparison metrics
    """
    logger.info("Comparing teacher and student models...")
    
    device = next(teacher_model.parameters()).device
    
    # Set models to evaluation mode
    teacher_model.eval()
    student_model.eval()
    
    # Initialize metrics
    results = {
        'teacher': {'accuracy': 0.0, 'size': 0.0, 'inference_time': 0.0},
        'student': {'accuracy': 0.0, 'size': 0.0, 'inference_time': 0.0}
    }
    
    # Calculate model sizes
    results['teacher']['size'] = get_model_size(teacher_model)
    results['student']['size'] = get_model_size(student_model)
    
    # Test accuracy
    for model_name, model in [('teacher', teacher_model), ('student', student_model)]:
        correct = 0
        total = 0
        total_time = 0.0
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f"Testing {model_name} model")
            for inputs, targets in test_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Measure inference time
                start_time = time.time()
                outputs = model(inputs)
                torch.cuda.synchronize()  # Wait for all kernels to finish
                end_time = time.time()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Accumulate inference time
                batch_time = end_time - start_time
                total_time += batch_time
                
                # Update progress bar
                test_pbar.set_postfix({'acc': 100. * correct / total})
        
        # Calculate metrics
        accuracy = 100. * correct / total
        avg_inference_time = (total_time / total) * 1000  # Convert to ms per sample
        
        results[model_name]['accuracy'] = accuracy
        results[model_name]['inference_time'] = avg_inference_time
    
    # Calculate improvement ratios
    size_reduction = results['teacher']['size'] / results['student']['size']
    speed_improvement = results['teacher']['inference_time'] / results['student']['inference_time']
    accuracy_retention = results['student']['accuracy'] / results['teacher']['accuracy'] * 100
    
    # Log results
    logger.info(f"Teacher model: Size={results['teacher']['size']:.2f}MB, "
               f"Accuracy={results['teacher']['accuracy']:.2f}%, "
               f"Inference time={results['teacher']['inference_time']:.2f}ms")
    
    logger.info(f"Student model: Size={results['student']['size']:.2f}MB, "
               f"Accuracy={results['student']['accuracy']:.2f}%, "
               f"Inference time={results['student']['inference_time']:.2f}ms")
    
    logger.info(f"Improvement: Size={size_reduction:.2f}x smaller, "
               f"Speed={speed_improvement:.2f}x faster, "
               f"Retained {accuracy_retention:.2f}% of accuracy")
    
    return results


def distill_knowledge(
    teacher_model_path: str,
    student_model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    alpha: float = 0.5,
    temperature: float = 2.0,
    epochs: int = 100,
    learning_rate: float = 0.01,
    test_loader: Optional[DataLoader] = None
) -> nn.Module:
    """
    Main function to perform knowledge distillation.
    
    Args:
        teacher_model_path (str): Path to the teacher model checkpoint
        student_model_name (str): Name of the student model architecture
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        alpha (float): Weight for the distillation loss
        temperature (float): Temperature for the soft targets
        epochs (int): Number of training epochs
        learning_rate (float): Initial learning rate
        test_loader (DataLoader, optional): Test data loader for evaluation
        
    Returns:
        nn.Module: Trained student model
    """
    # Load the teacher model
    logger.info(f"Loading teacher model from {teacher_model_path}")
    
    # Determine the number of classes from the model
    state_dict = torch.load(teacher_model_path)
    
    # Determine the number of classes from the output layer weights
    if 'fc.weight' in state_dict:  # ResNet architecture
        num_classes = state_dict['fc.weight'].size(0)
        teacher_model_name = 'resnet50'  # Assume ResNet50 by default
        
        # Check if it's another ResNet variant based on layer dimensions
        fc_in_features = state_dict['fc.weight'].size(1)
        if fc_in_features == 512:
            teacher_model_name = 'resnet18'
        elif fc_in_features == 1024:
            teacher_model_name = 'resnet34'
        elif fc_in_features == 2048:
            teacher_model_name = 'resnet50' or 'resnet101'  # Can't distinguish easily, assume ResNet50
            
    elif 'classifier.1.weight' in state_dict:  # MobileNet or EfficientNet
        num_classes = state_dict['classifier.1.weight'].size(0)
        if any('blocks' in key for key in state_dict.keys()):
            teacher_model_name = 'efficientnet'
        else:
            teacher_model_name = 'mobilenet'
    else:
        raise ValueError("Unable to determine model architecture from state dict")
    
    # Load the teacher model with the correct number of classes
    teacher_model, _, _ = load_baseline_model(teacher_model_name, num_classes)
    teacher_model.load_state_dict(state_dict)
    teacher_model.eval()  # Set to evaluation mode
    
    # Load the student model
    logger.info(f"Creating student model: {student_model_name}")
    student_model, _, _ = load_baseline_model(student_model_name, num_classes)
    
    # Log model sizes
    teacher_size = get_model_size(teacher_model)
    student_size = get_model_size(student_model)
    logger.info(f"Teacher model size: {teacher_size:.2f} MB")
    logger.info(f"Student model size: {student_size:.2f} MB (before training)")
    logger.info(f"Size reduction: {teacher_size/student_size:.2f}x")
    
    # Train the student model
    student_model = train_student_model(
        teacher_model, student_model, train_loader, val_loader,
        alpha, temperature, epochs, learning_rate
    )
    
    # Compare models if test loader is provided
    if test_loader is not None:
        _ = compare_models(teacher_model, student_model, test_loader)
    
    return student_model


if __name__ == '__main__':
    # Test the distillation module
    logging.basicConfig(level=logging.INFO)
    
    # For testing purposes, we'll just create example models
    logger.info("This module provides knowledge distillation functionality.")
    logger.info("Run the main pipeline to use this module with real data.") 