#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the model compression pipeline.
This script orchestrates the entire pipeline from data loading to model evaluation.
"""

import os
import sys
import argparse
import logging
import yaml
from datetime import datetime

# Add the 'src' directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import load_dataset, get_dataset_info
from src.models.baseline import load_baseline_model, train_baseline_model
from src.compression.pruning import prune_model
from src.compression.quantization import quantize_model
from src.compression.distillation import distill_knowledge
from src.compression.lottery_ticket import apply_lottery_ticket_pruning
from src.evaluation.benchmark import evaluate_model, save_metrics, generate_report
import torch


def setup_logging():
    """Set up logging configuration."""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Model Compression Pipeline')
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['baseline', 'prune', 'quantize', 'distill', 'lottery_ticket', 'evaluate', 'full', 'report'],
                        help='Mode of operation')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    
    # Baseline model arguments
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'mobilenet', 
                                'efficientnet', 'vit_b_16', 'vit_b_32', 'vit_l_16'],
                        help='Base model architecture')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet', 'flowers102'],
                        help='Dataset to use')
    
    # Pruning arguments
    parser.add_argument('--prune_rate', type=float, default=0.5,
                        help='Rate of weights to prune (0.0 to 1.0)')
    parser.add_argument('--prune_method', type=str, default='magnitude',
                        choices=['magnitude', 'random', 'structured'],
                        help='Method of pruning')
    
    # Lottery Ticket arguments
    parser.add_argument('--lottery_iterations', type=int, default=5,
                        help='Number of iterations for Lottery Ticket Hypothesis')
    parser.add_argument('--lottery_prune_percent', type=float, default=0.2,
                        help='Percentage of weights to prune in each Lottery Ticket iteration')
    
    # Quantization arguments
    parser.add_argument('--bits', type=int, default=8,
                        choices=[8, 4, 2],
                        help='Bit precision for quantization')
    parser.add_argument('--quantize_method', type=str, default='post_training',
                        choices=['post_training', 'quantization_aware'],
                        help='Method of quantization')
    
    # Distillation arguments
    parser.add_argument('--student', type=str, default='resnet18',
                        choices=['resnet18', 'mobilenet', 'efficientnet-b0'],
                        help='Student model architecture for distillation')
    parser.add_argument('--teacher', type=str, default=None,
                        help='Teacher model path (defaults to baseline if None)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for distillation loss (0.0 to 1.0)')
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='Temperature for soft targets in distillation')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate')
    
    # Evaluation arguments
    parser.add_argument('--metrics', type=str, nargs='+', 
                        default=['accuracy', 'size', 'latency', 'memory'],
                        help='Metrics to evaluate')
    
    # Report arguments
    parser.add_argument('--report_title', type=str, default=None,
                        help='Title for the comparison report')
    parser.add_argument('--report_dir', type=str, default=None,
                        help='Directory to save the report')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        logging.warning(f"Config file {config_path} not found. Using default parameters.")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def run_pipeline(args, config, logger):
    """Run the model compression pipeline based on arguments."""
    # Load dataset
    logger.info(f"Loading {args.dataset} dataset...")
    train_loader, val_loader, test_loader = load_dataset(args.dataset, args.batch_size)
    
    # Get dataset info to access num_classes
    dataset_info = get_dataset_info(args.dataset)
    num_classes = dataset_info['num_classes']
    
    # Base directory for saving models
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    baseline_path = os.path.join(results_dir, f"baseline_{args.model}_{args.dataset}.pth")
    
    # Mode: baseline - Train or load the baseline model
    if args.mode == 'baseline' or args.mode == 'full' or not os.path.exists(baseline_path):
        logger.info(f"Setting up baseline {args.model} model...")
        model, optimizer, criterion = load_baseline_model(args.model, num_classes)
        
        if not os.path.exists(baseline_path):
            logger.info("Training baseline model...")
            train_baseline_model(model, train_loader, val_loader, optimizer, criterion, 
                                args.epochs, args.learning_rate, baseline_path)
        else:
            logger.info(f"Loading pre-trained baseline model from {baseline_path}")
            model.load_state_dict(torch.load(baseline_path))
        
        if args.mode == 'baseline' or args.mode == 'full':
            logger.info("Evaluating baseline model...")
            baseline_metrics = evaluate_model(model, test_loader, args.metrics)
            save_metrics(baseline_metrics, os.path.join(results_dir, f"baseline_{args.model}_{args.dataset}_metrics.json"))
    
    # Mode: prune - Apply pruning to the baseline model
    if args.mode == 'prune' or args.mode == 'full':
        logger.info(f"Applying {args.prune_method} pruning with rate {args.prune_rate}...")
        pruned_model = prune_model(baseline_path, args.prune_rate, args.prune_method, 
                                  train_loader, val_loader, args.epochs)
        
        pruned_path = os.path.join(results_dir, f"pruned_{args.model}_{args.dataset}_{args.prune_method}_{args.prune_rate}.pth")
        torch.save(pruned_model.state_dict(), pruned_path)
        
        logger.info("Evaluating pruned model...")
        pruned_metrics = evaluate_model(pruned_model, test_loader, args.metrics)
        save_metrics(pruned_metrics, os.path.join(results_dir, f"pruned_{args.model}_{args.dataset}_{args.prune_method}_{args.prune_rate}_metrics.json"))
    
    # Mode: lottery_ticket - Apply Lottery Ticket Hypothesis pruning to the baseline model
    if args.mode == 'lottery_ticket' or args.mode == 'full':
        logger.info(f"Applying Lottery Ticket Hypothesis pruning with {args.lottery_iterations} iterations...")
        lottery_model = apply_lottery_ticket_pruning(
            baseline_path,
            train_loader,
            val_loader,
            prune_percent=args.lottery_prune_percent,
            iterations=args.lottery_iterations,
            epochs_per_iteration=20,  # Fewer epochs per iteration
            learning_rate=args.learning_rate
        )
        
        lottery_path = os.path.join(
            results_dir, 
            f"lottery_{args.model}_{args.dataset}_i{args.lottery_iterations}_p{args.lottery_prune_percent}.pth"
        )
        torch.save(lottery_model.state_dict(), lottery_path)
        
        logger.info("Evaluating lottery ticket model...")
        lottery_metrics = evaluate_model(lottery_model, test_loader, args.metrics)
        save_metrics(
            lottery_metrics, 
            os.path.join(
                results_dir, 
                f"lottery_{args.model}_{args.dataset}_i{args.lottery_iterations}_p{args.lottery_prune_percent}_metrics.json"
            )
        )
    
    # Mode: quantize - Apply quantization to the baseline or pruned model
    if args.mode == 'quantize' or args.mode == 'full':
        source_model_path = baseline_path
        if args.mode == 'full':
            source_model_path = os.path.join(results_dir, f"pruned_{args.model}_{args.dataset}_{args.prune_method}_{args.prune_rate}.pth")
        
        logger.info(f"Applying {args.quantize_method} quantization with {args.bits} bits...")
        quantized_model = quantize_model(source_model_path, args.bits, args.quantize_method, 
                                        test_loader)
        
        quantized_path = os.path.join(results_dir, 
                                     f"quantized_{args.model}_{args.dataset}_{args.quantize_method}_{args.bits}bits.pth")
        torch.save(quantized_model.state_dict(), quantized_path)
        
        logger.info("Evaluating quantized model...")
        quantized_metrics = evaluate_model(quantized_model, test_loader, args.metrics)
        save_metrics(quantized_metrics, os.path.join(results_dir, 
                                                   f"quantized_{args.model}_{args.dataset}_{args.quantize_method}_{args.bits}bits_metrics.json"))
    
    # Mode: distill - Apply knowledge distillation
    if args.mode == 'distill' or args.mode == 'full':
        teacher_path = args.teacher if args.teacher else baseline_path
        
        logger.info(f"Applying knowledge distillation from {teacher_path} to {args.student}...")
        student_model = distill_knowledge(teacher_path, args.student, train_loader, val_loader,
                                        args.alpha, args.temperature, args.epochs, args.learning_rate)
        
        distilled_path = os.path.join(results_dir, 
                                     f"distilled_{args.student}_from_{os.path.basename(teacher_path).split('.')[0]}.pth")
        torch.save(student_model.state_dict(), distilled_path)
        
        logger.info("Evaluating distilled model...")
        distilled_metrics = evaluate_model(student_model, test_loader, args.metrics)
        save_metrics(distilled_metrics, os.path.join(results_dir, 
                                                   f"distilled_{args.student}_from_{os.path.basename(teacher_path).split('.')[0]}_metrics.json"))
    
    # Mode: evaluate - Evaluate a specific model
    if args.mode == 'evaluate':
        if not args.model_path:
            logger.error("Model path is required for evaluation mode")
            return
        
        logger.info(f"Loading model from {args.model_path} for evaluation...")
        # Load the model based on the architecture
        model, _, _ = load_baseline_model(args.model, num_classes)
        model.load_state_dict(torch.load(args.model_path))
        
        logger.info("Evaluating model...")
        metrics = evaluate_model(model, test_loader, args.metrics)
        save_metrics(metrics, os.path.join(results_dir, f"{os.path.basename(args.model_path).split('.')[0]}_metrics.json"))
    
    # Mode: report - Generate a comparison report
    if args.mode == 'report':
        logger.info("Generating comparison report...")
        
        # Collect all metrics files
        metrics_files = []
        for filename in os.listdir(results_dir):
            if filename.endswith('_metrics.json'):
                metrics_files.append(os.path.join(results_dir, filename))
        
        if not metrics_files:
            logger.error("No metrics files found for report generation")
            return
        
        # Set up report directory
        report_dir = args.report_dir or os.path.join(results_dir, 'reports')
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate report
        report_title = args.report_title or f"Model Compression Comparison - {args.model} on {args.dataset}"
        generate_report(metrics_files, report_dir, title=report_title)
        
        logger.info(f"Report generated at {os.path.join(report_dir, 'compression_report.html')}")
    
    logger.info("Pipeline completed successfully!")


def main():
    """Main function."""
    # Set up logging
    logger = setup_logging()
    logger.info("Starting model compression pipeline...")
    
    # Parse arguments
    args = parse_arguments()
    logger.info(f"Running in {args.mode} mode")
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.config)
    config = load_config(config_path)
    
    # Run the pipeline
    try:
        run_pipeline(args, config, logger)
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 