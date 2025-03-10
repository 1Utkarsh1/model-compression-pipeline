#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation and benchmarking module for model comparison.
"""

import os
import json
import time
import logging
import psutil
import gc
from typing import Dict, List, Optional, Union, Any

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.compression.quantization import get_model_size

logger = logging.getLogger(__name__)


def get_model_accuracy(model: nn.Module, test_loader: DataLoader) -> float:
    """
    Evaluate model accuracy on the test set.
    
    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        
    Returns:
        float: Accuracy as a percentage
    """
    logger.info("Evaluating model accuracy...")
    
    device = next(model.parameters()).device
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating accuracy"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100.0 * correct / total
    logger.info(f"Model accuracy: {accuracy:.2f}%")
    
    return accuracy


def measure_inference_latency(
    model: nn.Module, 
    test_loader: DataLoader, 
    num_batches: int = 100,
    warmup_batches: int = 10
) -> Dict[str, float]:
    """
    Measure inference latency of the model.
    
    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        num_batches (int): Number of batches to measure
        warmup_batches (int): Number of warmup batches
        
    Returns:
        dict: Latency metrics in milliseconds
    """
    logger.info("Measuring inference latency...")
    
    device = next(model.parameters()).device
    model.eval()
    
    # Warmup
    logger.info(f"Warming up for {warmup_batches} batches...")
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= warmup_batches:
                break
            inputs = inputs.to(device)
            _ = model(inputs)
    
    # Measure batch latency
    logger.info(f"Measuring latency over {num_batches} batches...")
    batch_times = []
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= num_batches:
                break
                
            inputs = inputs.to(device)
            
            # Synchronize to make sure GPU operations are done
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(inputs)
            
            # Synchronize again
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            end_time = time.time()
            
            # Convert to milliseconds
            batch_times.append((end_time - start_time) * 1000)
    
    # Calculate statistics
    batch_times = np.array(batch_times)
    mean_latency = np.mean(batch_times)
    p50_latency = np.percentile(batch_times, 50)
    p95_latency = np.percentile(batch_times, 95)
    
    # Calculate per-sample latency
    batch_size = test_loader.batch_size
    per_sample_latency = mean_latency / batch_size
    
    latency_metrics = {
        'batch_mean_ms': mean_latency,
        'batch_p50_ms': p50_latency,
        'batch_p95_ms': p95_latency,
        'per_sample_ms': per_sample_latency
    }
    
    logger.info(f"Mean batch latency: {mean_latency:.2f} ms")
    logger.info(f"P50 batch latency: {p50_latency:.2f} ms")
    logger.info(f"P95 batch latency: {p95_latency:.2f} ms")
    logger.info(f"Per-sample latency: {per_sample_latency:.2f} ms")
    
    return latency_metrics


def measure_memory_usage(
    model: nn.Module, 
    test_loader: DataLoader, 
    num_samples: int = 100
) -> Dict[str, float]:
    """
    Measure memory usage of the model during inference.
    
    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        num_samples (int): Number of samples to measure
        
    Returns:
        dict: Memory usage metrics in MB
    """
    logger.info("Measuring memory usage...")
    
    device = next(model.parameters()).device
    model.eval()
    
    # Force garbage collection
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Get baseline memory usage
    process = psutil.Process(os.getpid())
    baseline_memory = process.memory_info().rss / (1024 * 1024)  # in MB
    
    if device.type == 'cuda':
        baseline_gpu_memory = torch.cuda.memory_allocated(device) / (1024 * 1024)  # in MB
    else:
        baseline_gpu_memory = 0
    
    # Run inference and measure peak memory
    peak_cpu_memory = baseline_memory
    peak_gpu_memory = baseline_gpu_memory
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i * test_loader.batch_size >= num_samples:
                break
                
            inputs = inputs.to(device)
            _ = model(inputs)
            
            # Update peak memory
            current_cpu_memory = process.memory_info().rss / (1024 * 1024)
            peak_cpu_memory = max(peak_cpu_memory, current_cpu_memory)
            
            if device.type == 'cuda':
                current_gpu_memory = torch.cuda.memory_allocated(device) / (1024 * 1024)
                peak_gpu_memory = max(peak_gpu_memory, current_gpu_memory)
    
    # Calculate memory consumption
    cpu_memory_usage = peak_cpu_memory - baseline_memory
    gpu_memory_usage = peak_gpu_memory - baseline_gpu_memory
    
    memory_metrics = {
        'cpu_memory_mb': cpu_memory_usage,
        'gpu_memory_mb': gpu_memory_usage,
        'total_memory_mb': cpu_memory_usage + gpu_memory_usage
    }
    
    logger.info(f"CPU memory usage: {cpu_memory_usage:.2f} MB")
    logger.info(f"GPU memory usage: {gpu_memory_usage:.2f} MB")
    logger.info(f"Total memory usage: {memory_metrics['total_memory_mb']:.2f} MB")
    
    return memory_metrics


def evaluate_model(
    model: nn.Module, 
    test_loader: DataLoader, 
    metrics: List[str] = ['accuracy', 'size', 'latency', 'memory']
) -> Dict[str, Any]:
    """
    Evaluate a model on multiple metrics.
    
    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        metrics (list): List of metrics to evaluate
        
    Returns:
        dict: Evaluation results for each metric
    """
    logger.info(f"Evaluating model on metrics: {metrics}")
    
    results = {}
    
    # Accuracy
    if 'accuracy' in metrics:
        results['accuracy'] = get_model_accuracy(model, test_loader)
    
    # Model size
    if 'size' in metrics:
        model_size_mb = get_model_size(model)
        results['size_mb'] = model_size_mb
        logger.info(f"Model size: {model_size_mb:.2f} MB")
    
    # Inference latency
    if 'latency' in metrics:
        results['latency'] = measure_inference_latency(model, test_loader)
    
    # Memory usage
    if 'memory' in metrics:
        results['memory'] = measure_memory_usage(model, test_loader)
    
    return results


def save_metrics(metrics: Dict[str, Any], save_path: str) -> None:
    """
    Save metrics to a JSON file.
    
    Args:
        metrics (dict): Metrics to save
        save_path (str): Path to save the metrics
        
    Returns:
        None
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert any numpy types to Python types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    converted_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            converted_metrics[key] = {k: convert_for_json(v) for k, v in value.items()}
        else:
            converted_metrics[key] = convert_for_json(value)
    
    with open(save_path, 'w') as f:
        json.dump(converted_metrics, f, indent=4)
    
    logger.info(f"Metrics saved to {save_path}")


def load_metrics(metrics_path: str) -> Dict[str, Any]:
    """
    Load metrics from a JSON file.
    
    Args:
        metrics_path (str): Path to the metrics file
        
    Returns:
        dict: Loaded metrics
    """
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def compare_models_metrics(
    metrics_files: List[str],
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare metrics across multiple models and generate visualizations.
    
    Args:
        metrics_files (list): List of paths to metrics files
        output_dir (str, optional): Directory to save visualizations
        
    Returns:
        DataFrame: Comparison of metrics across models
    """
    logger.info(f"Comparing metrics from {len(metrics_files)} models")
    
    # Load metrics for each model
    all_metrics = []
    model_names = []
    
    for metrics_file in metrics_files:
        metrics = load_metrics(metrics_file)
        model_name = os.path.basename(metrics_file).split('_metrics.json')[0]
        model_names.append(model_name)
        
        # Extract key metrics into a flat dictionary
        model_metrics = {
            'model': model_name,
            'accuracy': metrics.get('accuracy', float('nan')),
            'size_mb': metrics.get('size_mb', float('nan')),
            'latency_ms': metrics.get('latency', {}).get('per_sample_ms', float('nan')),
            'memory_mb': metrics.get('memory', {}).get('total_memory_mb', float('nan'))
        }
        
        all_metrics.append(model_metrics)
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Set model as index
    df.set_index('model', inplace=True)
    
    # Display comparison
    logger.info("Model comparison:")
    logger.info("\n" + str(df))
    
    # Generate visualizations if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Reset index for plotting
        df_plot = df.reset_index()
        
        # Accuracy vs. Size plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='size_mb', y='accuracy', hue='model', s=100, data=df_plot)
        plt.title('Accuracy vs. Model Size')
        plt.xlabel('Model Size (MB)')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'accuracy_vs_size.png'), dpi=300, bbox_inches='tight')
        
        # Accuracy vs. Latency plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='latency_ms', y='accuracy', hue='model', s=100, data=df_plot)
        plt.title('Accuracy vs. Inference Latency')
        plt.xlabel('Inference Latency (ms/sample)')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'accuracy_vs_latency.png'), dpi=300, bbox_inches='tight')
        
        # Size vs. Memory plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='size_mb', y='memory_mb', hue='model', s=100, data=df_plot)
        plt.title('Model Size vs. Memory Usage')
        plt.xlabel('Model Size (MB)')
        plt.ylabel('Memory Usage (MB)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'size_vs_memory.png'), dpi=300, bbox_inches='tight')
        
        # Bar plot of all metrics (normalized)
        plt.figure(figsize=(12, 8))
        
        # Normalize metrics for visualization
        df_norm = df.copy()
        for col in df_norm.columns:
            if col != 'model':
                max_val = df_norm[col].max()
                if max_val > 0:
                    df_norm[col] = df_norm[col] / max_val
        
        df_norm = df_norm.reset_index()
        df_norm_melt = pd.melt(df_norm, id_vars=['model'], var_name='Metric', value_name='Normalized Value')
        
        sns.barplot(x='model', y='Normalized Value', hue='Metric', data=df_norm_melt)
        plt.title('Normalized Metrics Comparison (Higher is Better for Accuracy, Lower is Better for Others)')
        plt.ylabel('Normalized Value')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.legend(title='Metric')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'normalized_metrics.png'), dpi=300, bbox_inches='tight')
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    return df


def generate_report(
    metrics_files: List[str],
    output_dir: str,
    title: str = "Model Compression Comparison Report"
) -> None:
    """
    Generate a comprehensive HTML report comparing models.
    
    Args:
        metrics_files (list): List of paths to metrics files
        output_dir (str): Directory to save the report
        title (str): Title of the report
        
    Returns:
        None
    """
    logger.info("Generating comparison report...")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Compare models and generate visualizations
    df = compare_models_metrics(metrics_files, output_dir)
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 20px;
                color: #333;
            }}
            h1, h2 {{
                color: #2c3e50;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: right;
            }}
            th {{
                background-color: #f2f2f2;
                text-align: center;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .image-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
            }}
            .image-box {{
                width: 48%;
                margin-bottom: 20px;
            }}
            img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
            }}
            .metric-highlight {{
                font-weight: bold;
            }}
            .best-value {{
                color: #27ae60;
                font-weight: bold;
            }}
            .percentage {{
                color: #d35400;
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        
        <h2>Model Comparison Summary</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Accuracy (%)</th>
                <th>Size (MB)</th>
                <th>Latency (ms/sample)</th>
                <th>Memory (MB)</th>
            </tr>
    """
    
    # Add table rows for each model
    for model, row in df.iterrows():
        html_content += f"""
            <tr>
                <td>{model}</td>
                <td>{row['accuracy']:.2f}</td>
                <td>{row['size_mb']:.2f}</td>
                <td>{row['latency_ms']:.2f}</td>
                <td>{row['memory_mb']:.2f}</td>
            </tr>
        """
    
    # Calculate improvement metrics from baseline (assuming first model is baseline)
    baseline_model = df.index[0]
    baseline_metrics = df.loc[baseline_model]
    
    html_content += """
        </table>
        
        <h2>Improvement from Baseline</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Accuracy Retention</th>
                <th>Size Reduction</th>
                <th>Latency Improvement</th>
                <th>Memory Reduction</th>
            </tr>
    """
    
    for model, row in df.iterrows():
        if model == baseline_model:
            continue
            
        accuracy_retention = (row['accuracy'] / baseline_metrics['accuracy']) * 100
        size_reduction = baseline_metrics['size_mb'] / row['size_mb']
        latency_improvement = baseline_metrics['latency_ms'] / row['latency_ms']
        memory_reduction = baseline_metrics['memory_mb'] / row['memory_mb']
        
        html_content += f"""
            <tr>
                <td>{model}</td>
                <td>{accuracy_retention:.2f}%</td>
                <td>{size_reduction:.2f}x</td>
                <td>{latency_improvement:.2f}x</td>
                <td>{memory_reduction:.2f}x</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Visualizations</h2>
        <div class="image-container">
            <div class="image-box">
                <h3>Accuracy vs. Model Size</h3>
                <img src="accuracy_vs_size.png" alt="Accuracy vs. Size">
            </div>
            <div class="image-box">
                <h3>Accuracy vs. Inference Latency</h3>
                <img src="accuracy_vs_latency.png" alt="Accuracy vs. Latency">
            </div>
            <div class="image-box">
                <h3>Model Size vs. Memory Usage</h3>
                <img src="size_vs_memory.png" alt="Size vs. Memory">
            </div>
            <div class="image-box">
                <h3>Normalized Metrics Comparison</h3>
                <img src="normalized_metrics.png" alt="Normalized Metrics">
            </div>
        </div>
        
        <h2>Conclusion</h2>
        <p>
            This report compares the performance of different model compression techniques against the baseline model.
            The key metrics considered are accuracy, model size, inference latency, and memory usage.
        </p>
        <p>
            <strong>Key Findings:</strong>
        </p>
        <ul>
    """
    
    # Add some automated findings
    # Find best model for size reduction
    best_size_model = df['size_mb'].idxmin()
    size_reduction = baseline_metrics['size_mb'] / df.loc[best_size_model, 'size_mb']
    accuracy_drop = baseline_metrics['accuracy'] - df.loc[best_size_model, 'accuracy']
    
    html_content += f"""
            <li>The <span class="metric-highlight">{best_size_model}</span> model provides the best size reduction at 
                <span class="best-value">{size_reduction:.2f}x</span> smaller than the baseline, 
                with an accuracy drop of <span class="percentage">{accuracy_drop:.2f}%</span>.</li>
    """
    
    # Find best model for latency
    best_latency_model = df['latency_ms'].idxmin()
    latency_improvement = baseline_metrics['latency_ms'] / df.loc[best_latency_model, 'latency_ms']
    accuracy_drop = baseline_metrics['accuracy'] - df.loc[best_latency_model, 'accuracy']
    
    html_content += f"""
            <li>The <span class="metric-highlight">{best_latency_model}</span> model provides the best inference speed at 
                <span class="best-value">{latency_improvement:.2f}x</span> faster than the baseline, 
                with an accuracy drop of <span class="percentage">{accuracy_drop:.2f}%</span>.</li>
    """
    
    # Find best accuracy retention excluding baseline
    df_no_baseline = df.drop(baseline_model)
    if not df_no_baseline.empty:
        best_accuracy_model = df_no_baseline['accuracy'].idxmax()
        accuracy_retention = (df.loc[best_accuracy_model, 'accuracy'] / baseline_metrics['accuracy']) * 100
        size_reduction = baseline_metrics['size_mb'] / df.loc[best_accuracy_model, 'size_mb']
        
        html_content += f"""
                <li>The <span class="metric-highlight">{best_accuracy_model}</span> model provides the best accuracy retention at 
                    <span class="best-value">{accuracy_retention:.2f}%</span> of the baseline, 
                    while still being <span class="percentage">{size_reduction:.2f}x</span> smaller.</li>
        """
    
    html_content += """
        </ul>
        <p>
            For detailed analysis and implementation details, please refer to the accompanying notebooks and source code.
        </p>
    </body>
    </html>
    """
    
    # Write HTML file
    report_path = os.path.join(output_dir, 'compression_report.html')
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Report generated and saved to {report_path}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Evaluation module for benchmarking models.")
    logger.info("Run the main pipeline to use this module with real models.") 