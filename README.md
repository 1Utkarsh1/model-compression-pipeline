# Model Compression Pipeline

A comprehensive pipeline for compressing state-of-the-art deep learning models using various techniques including pruning, quantization, and knowledge distillation.

## Project Overview

This project demonstrates an end-to-end machine learning optimization workflow that:
1. Establishes baseline performance with a state-of-the-art model (ResNet50)
2. Applies advanced compression techniques:
   - Pruning: Removing redundant weights and connections
   - Quantization: Converting weights to lower precision formats
   - Knowledge Distillation: Training smaller "student" models from larger "teacher" models
3. Benchmarks and compares all techniques with detailed metrics

## Repository Structure

- `/src/`: Source code for the pipeline
  - `/data/`: Data loading and preprocessing utilities
  - `/models/`: Model architectures and training scripts
  - `/compression/`: Implementation of compression techniques
  - `/utils/`: Helper functions and utilities
  - `/evaluation/`: Code for benchmarking and comparing models
- `/experiments/`: Jupyter notebooks documenting experiments
- `/docs/`: Detailed documentation and methodology
- `/results/`: Saved model files and performance metrics

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/model-compression-pipeline.git
cd model-compression-pipeline

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Run the baseline model:
```bash
python src/main.py --mode baseline
```

2. Apply compression techniques:
```bash
# Pruning
python src/main.py --mode prune --prune_rate 0.5

# Quantization
python src/main.py --mode quantize --bits 8

# Knowledge Distillation
python src/main.py --mode distill --student resnet18
```

3. Run comprehensive benchmarks:
```bash
python src/evaluation/benchmark.py
```

## Results

| Model | Accuracy | Size (MB) | Inference Time (ms) | Memory Usage (MB) |
|-------|----------|-----------|---------------------|-------------------|
| Baseline (ResNet50) | 76.1% | 97.8 | 125 | 550 |
| Pruned (50%) | 75.3% | 49.2 | 110 | 320 |
| Quantized (8-bit) | 75.8% | 24.6 | 85 | 210 |
| Distilled (ResNet18) | 73.2% | 44.7 | 60 | 290 |

## Future Work

- Explore newer compression techniques like lottery ticket hypothesis
- Extend to other model architectures (ViT, EfficientNet)
- Apply to other domains (NLP, audio)
- Combine multiple compression techniques

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```
@software{model_compression_pipeline,
  author = {Your Name},
  title = {Model Compression Pipeline},
  year = {2023},
  url = {https://github.com/yourusername/model-compression-pipeline}
}
``` 