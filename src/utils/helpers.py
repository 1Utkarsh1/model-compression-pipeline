#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper utilities for the model compression pipeline.
"""

import os
import sys
import time
import logging
import json
from typing import Dict, Any, List, Optional, Union

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tabulate import tabulate

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str, name: str = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_dir (str): Directory to save log files
        name (str, optional): Logger name
        
    Returns:
        logging.Logger: Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'{name or "pipeline"}_{timestamp}.log')
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Return the requested logger
    return logging.getLogger(name)


def save_dict_to_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save a dictionary to a JSON file.
    
    Args:
        data (dict): Dictionary to save
        filepath (str): Path to save the JSON file
        
    Returns:
        None
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy types to Python types
    def convert_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Convert all values in the dictionary
    processed_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            processed_data[key] = {k: convert_for_json(v) for k, v in value.items()}
        else:
            processed_data[key] = convert_for_json(value)
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(processed_data, f, indent=4)


def load_dict_from_json(filepath: str) -> Dict[str, Any]:
    """
    Load a dictionary from a JSON file.
    
    Args:
        filepath (str): Path to the JSON file
        
    Returns:
        dict: Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def print_model_summary(model: torch.nn.Module) -> None:
    """
    Print a summary of the model architecture and parameters.
    
    Args:
        model (torch.nn.Module): PyTorch model
        
    Returns:
        None
    """
    # Get model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Print summary header
    print(f"\nModel Summary: {model.__class__.__name__}")
    print("=" * 80)
    
    # Print parameter counts
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Print layer information
    print("\nLayer Information:")
    print("-" * 80)
    
    table_data = []
    for name, module in model.named_modules():
        if not name:  # Skip the root module
            continue
        
        # Get parameter count for the module
        module_params = sum(p.numel() for p in module.parameters())
        
        # Skip if no parameters (e.g., ReLU, Dropout)
        if module_params == 0:
            continue
        
        # Add row to table
        table_data.append([
            name,
            module.__class__.__name__,
            module_params,
            f"{module_params / total_params * 100:.2f}%"
        ])
    
    # Print table
    print(tabulate(
        table_data,
        headers=["Name", "Type", "Parameters", "% of Total"],
        tablefmt="grid"
    ))
    
    print("=" * 80)


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None) -> Figure:
    """
    Plot training history metrics.
    
    Args:
        history (dict): Dictionary containing training history metrics
        save_path (str, optional): Path to save the plot
        
    Returns:
        Figure: Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training and validation loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot training and validation accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def calculate_model_memory_footprint(model: torch.nn.Module) -> Dict[str, float]:
    """
    Estimate the memory footprint of a model.
    
    Args:
        model (torch.nn.Module): PyTorch model
        
    Returns:
        dict: Memory usage in different formats
    """
    # Get total number of parameters
    param_count = sum(p.numel() for p in model.parameters())
    
    # Estimate memory usage (assuming float32)
    param_size_bytes = param_count * 4  # 4 bytes per float32 parameter
    param_size_mb = param_size_bytes / (1024 * 1024)
    
    # Get optimizer overhead (assuming Adam: 3x model size)
    optimizer_overhead_mb = param_size_mb * 3
    
    # Get gradient memory (same as model size)
    gradient_memory_mb = param_size_mb
    
    # Estimate forward pass memory (very rough approximation)
    forward_memory_mb = param_size_mb * 2
    
    # Total training memory
    total_training_mb = param_size_mb + optimizer_overhead_mb + gradient_memory_mb + forward_memory_mb
    
    # Total inference memory
    total_inference_mb = param_size_mb + forward_memory_mb
    
    return {
        "parameters": param_count,
        "model_size_mb": param_size_mb,
        "optimizer_overhead_mb": optimizer_overhead_mb,
        "gradient_memory_mb": gradient_memory_mb,
        "forward_memory_mb": forward_memory_mb,
        "total_training_mb": total_training_mb,
        "total_inference_mb": total_inference_mb
    }


def get_optimal_batch_size(model: torch.nn.Module, input_shape: tuple, max_memory_gb: float = 12.0) -> int:
    """
    Estimate the optimal batch size based on available memory.
    
    Args:
        model (torch.nn.Module): PyTorch model
        input_shape (tuple): Shape of a single input sample (e.g., (3, 224, 224))
        max_memory_gb (float): Maximum GPU memory in GB
        
    Returns:
        int: Estimated optimal batch size
    """
    # Estimate memory required per sample
    sample_size_bytes = np.prod(input_shape) * 4  # 4 bytes per float32
    
    # Get model memory footprint
    model_memory = calculate_model_memory_footprint(model)
    
    # Convert max memory to bytes
    max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
    
    # Memory available for batches (subtract model memory)
    available_memory = max_memory_bytes - (model_memory["total_training_mb"] * 1024 * 1024)
    
    # Estimate batch size (with safety factor of 0.7)
    max_batch_size = int((available_memory * 0.7) / sample_size_bytes)
    
    # Ensure batch size is at least 1
    max_batch_size = max(1, max_batch_size)
    
    # Round to the nearest power of 2 (common practice)
    power_of_2 = 2 ** int(np.log2(max_batch_size))
    
    return power_of_2


if __name__ == "__main__":
    # Test the utilities
    logger = setup_logging("./logs", "test")
    logger.info("Testing helper utilities...") 