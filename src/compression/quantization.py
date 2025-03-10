#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model quantization module for applying quantization techniques.
"""

import os
import logging
import time
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.quantization import (
    get_default_qconfig,
    quantize_jit,
    prepare_jit,
    convert_jit,
    QuantStub,
    DeQuantStub,
    QConfig,
    prepare,
    convert,
    default_qconfig,
    quantize
)
from tqdm import tqdm

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.baseline import load_baseline_model

logger = logging.getLogger(__name__)


class QuantizableModel(nn.Module):
    """
    Wrapper class for making models quantizable.
    """
    def __init__(self, model: nn.Module):
        super(QuantizableModel, self).__init__()
        self.model = model
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


def get_model_size(model: nn.Module) -> float:
    """
    Calculate the model size in MB.
    
    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        float: Model size in MB
    """
    # Save the model temporarily to get its size
    temp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                            'results', 'temp_model_size.pth')
    torch.save(model.state_dict(), temp_path)
    
    # Get the file size in MB
    size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    
    # Clean up
    os.remove(temp_path)
    
    return size_mb


def apply_post_training_static_quantization(
    model: nn.Module,
    calibration_loader: DataLoader,
    bits: int = 8
) -> nn.Module:
    """
    Apply post-training static quantization to the model.
    
    Args:
        model (nn.Module): Model to quantize
        calibration_loader (DataLoader): DataLoader for calibration
        bits (int): Bit precision for quantization (8, 4, or 2)
        
    Returns:
        nn.Module: Quantized model
    """
    logger.info(f"Applying post-training static quantization with {bits} bits")
    
    # Prepare for quantization
    quantizable_model = QuantizableModel(model)
    
    # Set the qconfig based on the bit precision
    if bits == 8:
        qconfig = default_qconfig
    elif bits == 4:
        # 4-bit quantization (reduced precision)
        qconfig = QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(
                qscheme=torch.per_tensor_symmetric, dtype=torch.quint4x2
            ),
            weight=torch.quantization.MinMaxObserver.with_args(
                qscheme=torch.per_tensor_symmetric, dtype=torch.qint4x2
            )
        )
    elif bits == 2:
        # 2-bit quantization (highly reduced precision)
        qconfig = QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(
                qscheme=torch.per_tensor_symmetric, dtype=torch.quint2x4
            ),
            weight=torch.quantization.MinMaxObserver.with_args(
                qscheme=torch.per_tensor_symmetric, dtype=torch.qint2x4
            )
        )
    else:
        raise ValueError(f"Unsupported bit precision: {bits}")
    
    quantizable_model.qconfig = qconfig
    
    # Prepare model for calibration
    model_prepared = prepare(quantizable_model)
    
    # Calibrate with the data
    logger.info("Calibrating with data...")
    model_prepared.eval()
    
    device = next(model_prepared.parameters()).device
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(tqdm(calibration_loader, desc="Calibration")):
            inputs = inputs.to(device)
            model_prepared(inputs)
            
            # Use a subset for faster calibration
            if batch_idx >= 100:
                break
    
    # Convert the model to a quantized version
    quantized_model = convert(model_prepared)
    
    # Check model size before and after quantization
    orig_size = get_model_size(model)
    quant_size = get_model_size(quantized_model)
    compression_ratio = orig_size / quant_size if quant_size > 0 else float('inf')
    
    logger.info(f"Original model size: {orig_size:.2f} MB")
    logger.info(f"Quantized model size: {quant_size:.2f} MB")
    logger.info(f"Compression ratio: {compression_ratio:.2f}x")
    
    return quantized_model


def apply_quantization_aware_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    bits: int = 8,
    epochs: int = 5,
    learning_rate: float = 0.0001
) -> nn.Module:
    """
    Apply quantization-aware training to the model.
    
    Args:
        model (nn.Module): Model to quantize
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        bits (int): Bit precision for quantization (8, 4, or 2)
        epochs (int): Number of QAT epochs
        learning_rate (float): Learning rate for QAT
        
    Returns:
        nn.Module: Quantized model
    """
    logger.info(f"Applying quantization-aware training with {bits} bits for {epochs} epochs")
    
    # Prepare for quantization
    quantizable_model = QuantizableModel(model)
    
    # Set the qconfig based on the bit precision
    if bits == 8:
        qconfig = default_qconfig
    elif bits == 4:
        qconfig = QConfig(
            activation=torch.quantization.FakeQuantize.with_args(
                observer=torch.quantization.MovingAverageMinMaxObserver,
                quant_min=0, quant_max=15
            ),
            weight=torch.quantization.FakeQuantize.with_args(
                observer=torch.quantization.MovingAverageMinMaxObserver,
                quant_min=-8, quant_max=7
            )
        )
    elif bits == 2:
        qconfig = QConfig(
            activation=torch.quantization.FakeQuantize.with_args(
                observer=torch.quantization.MovingAverageMinMaxObserver,
                quant_min=0, quant_max=3
            ),
            weight=torch.quantization.FakeQuantize.with_args(
                observer=torch.quantization.MovingAverageMinMaxObserver,
                quant_min=-2, quant_max=1
            )
        )
    else:
        raise ValueError(f"Unsupported bit precision: {bits}")
    
    quantizable_model.qconfig = qconfig
    
    # Prepare model for QAT
    qat_model = prepare(quantizable_model, inplace=True)
    
    # Set up optimizer and criterion
    optimizer = torch.optim.Adam(qat_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Train the model with quantization awareness
    device = next(qat_model.parameters()).device
    
    best_val_acc = 0.0
    temp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                            'results', 'temp_qat_model.pth')
    
    for epoch in range(epochs):
        # Training phase
        qat_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"QAT Epoch {epoch+1}/{epochs} [Train]")
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = qat_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': train_loss / train_total,
                'acc': 100. * train_correct / train_total
            })
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        qat_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"QAT Epoch {epoch+1}/{epochs} [Valid]")
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = qat_model(inputs)
                loss = criterion(outputs, targets)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': val_loss / val_total,
                    'acc': 100. * val_correct / val_total
                })
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100. * val_correct / val_total
        
        logger.info(f"QAT Epoch {epoch+1}/{epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save the best model
        if val_acc > best_val_acc:
            logger.info(f"Validation accuracy improved from {best_val_acc:.2f}% to {val_acc:.2f}%")
            best_val_acc = val_acc
            torch.save(qat_model.state_dict(), temp_path)
    
    # Load the best model
    qat_model.load_state_dict(torch.load(temp_path))
    
    # Convert the QAT model to a quantized model
    quantized_model = convert(qat_model.eval(), inplace=False)
    
    # Clean up temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    # Check model size before and after quantization
    orig_size = get_model_size(model)
    quant_size = get_model_size(quantized_model)
    compression_ratio = orig_size / quant_size if quant_size > 0 else float('inf')
    
    logger.info(f"Original model size: {orig_size:.2f} MB")
    logger.info(f"Quantized model size: {quant_size:.2f} MB")
    logger.info(f"Compression ratio: {compression_ratio:.2f}x")
    
    return quantized_model


def measure_inference_speed(model: nn.Module, test_loader: DataLoader, num_batches: int = 50) -> float:
    """
    Measure the inference speed of a model in milliseconds per image.
    
    Args:
        model (nn.Module): Model to test
        test_loader (DataLoader): Test data loader
        num_batches (int): Number of batches to measure
        
    Returns:
        float: Average inference time per image in milliseconds
    """
    logger.info("Measuring inference speed...")
    
    device = next(model.parameters()).device
    model.eval()
    batch_size = test_loader.batch_size
    
    # Warm up the GPU
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            _ = model(inputs)
            if i >= 10:
                break
    
    # Measure inference time
    total_time = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= num_batches:
                break
            
            inputs = inputs.to(device)
            
            # Measure time
            start_time = time.time()
            _ = model(inputs)
            torch.cuda.synchronize()  # Wait for all kernels to finish
            end_time = time.time()
            
            # Accumulate time
            batch_time = end_time - start_time
            total_time += batch_time
            total_samples += inputs.size(0)
    
    # Calculate average inference time per image in milliseconds
    avg_time_ms = (total_time / total_samples) * 1000
    
    logger.info(f"Average inference time: {avg_time_ms:.2f} ms per image")
    
    return avg_time_ms


def quantize_model(
    model_path: str,
    bits: int = 8,
    method: str = 'post_training',
    calibration_loader: Optional[DataLoader] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 5
) -> nn.Module:
    """
    Main function to quantize a model.
    
    Args:
        model_path (str): Path to the model checkpoint
        bits (int): Bit precision for quantization (8, 4, or 2)
        method (str): Quantization method ('post_training' or 'quantization_aware')
        calibration_loader (DataLoader, optional): DataLoader for calibration (post-training)
        train_loader (DataLoader, optional): Training data loader (quantization-aware)
        val_loader (DataLoader, optional): Validation data loader (quantization-aware)
        epochs (int): Number of epochs for QAT
        
    Returns:
        nn.Module: Quantized model
    """
    # Load the model
    logger.info(f"Loading model from {model_path}")
    
    # We need to determine the number of classes from the model
    state_dict = torch.load(model_path)
    
    # Determine the number of classes from the output layer weights
    if 'fc.weight' in state_dict:  # ResNet architecture
        num_classes = state_dict['fc.weight'].size(0)
        model_name = 'resnet50'  # Assume ResNet50 by default
        
        # Check if it's another ResNet variant based on layer dimensions
        fc_in_features = state_dict['fc.weight'].size(1)
        if fc_in_features == 512:
            model_name = 'resnet18'
        elif fc_in_features == 1024:
            model_name = 'resnet34'
        elif fc_in_features == 2048:
            model_name = 'resnet50' or 'resnet101'  # Can't distinguish easily, assume ResNet50
            
    elif 'classifier.1.weight' in state_dict:  # MobileNet or EfficientNet
        num_classes = state_dict['classifier.1.weight'].size(0)
        if any('blocks' in key for key in state_dict.keys()):
            model_name = 'efficientnet'
        else:
            model_name = 'mobilenet'
    else:
        raise ValueError("Unable to determine model architecture from state dict")
    
    # Load the actual model with the correct number of classes
    model, _, _ = load_baseline_model(model_name, num_classes)
    model.load_state_dict(state_dict)
    
    # Save original model size for comparison
    orig_size = get_model_size(model)
    logger.info(f"Original model size: {orig_size:.2f} MB")
    
    # Apply the appropriate quantization method
    if method.lower() == 'post_training':
        if calibration_loader is None:
            if train_loader is not None:
                logger.info("Using training data for calibration")
                calibration_loader = train_loader
            else:
                raise ValueError("Calibration data loader is required for post-training quantization")
        
        quantized_model = apply_post_training_static_quantization(
            model, calibration_loader, bits
        )
    
    elif method.lower() == 'quantization_aware':
        if train_loader is None or val_loader is None:
            raise ValueError("Training and validation data loaders are required for quantization-aware training")
        
        quantized_model = apply_quantization_aware_training(
            model, train_loader, val_loader, bits, epochs
        )
    
    else:
        raise ValueError(f"Unsupported quantization method: {method}")
    
    # Measure inference speed (if a loader is available)
    if calibration_loader is not None:
        _ = measure_inference_speed(model, calibration_loader)
        _ = measure_inference_speed(quantized_model, calibration_loader)
    
    return quantized_model


if __name__ == '__main__':
    # Test the quantization module
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Load a small model for testing
    model, _, _ = load_baseline_model('resnet18', 10)
    
    # Save the model temporarily
    temp_path = 'temp_model.pth'
    torch.save(model.state_dict(), temp_path)
    
    # For testing purposes, we won't use actual data loaders
    # In a real scenario, you would load data loaders as needed
    
    # Clean up temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path) 