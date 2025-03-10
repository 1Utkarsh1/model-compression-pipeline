# 🚀 Model Compression Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

**A comprehensive framework for optimizing deep learning models through advanced compression techniques**

[Key Features](#key-features) •
[Architecture](#architecture) •
[Installation](#installation) •
[Usage](#usage) •
[Results](#results) •
[Roadmap](#roadmap) •
[Contributing](#contributing)

</div>

## 📑 Overview

The Model Compression Pipeline is an end-to-end framework designed to compress state-of-the-art deep learning models while preserving accuracy. This project implements various compression techniques including pruning, quantization, and knowledge distillation, allowing researchers and practitioners to optimize models for deployment on resource-constrained devices.

<div align="center">
    <img src="https://via.placeholder.com/800x400?text=Model+Compression+Workflow" alt="Model Compression Workflow" width="80%">
</div>

## ✨ Key Features

- **Modular Architecture**: Easily extensible framework for experimenting with different compression techniques
- **Multiple Compression Techniques**:
  - 🔪 **Pruning**: Remove redundant weights and connections
  - 🔢 **Quantization**: Convert weights to lower precision formats
  - 🧠 **Knowledge Distillation**: Train smaller "student" models from larger "teacher" models
  - 🎫 **Lottery Ticket Hypothesis**: Find and train sparse subnetworks with initial weights
- **Model Support**:
  - CNN architectures (ResNet, MobileNet, EfficientNet)
  - Vision Transformers (ViT variants)
- **Dataset Integration**:
  - CIFAR-10/100, ImageNet, Oxford Flowers102
- **Comprehensive Evaluation**:
  - Accuracy, model size, inference latency, memory usage benchmarks
  - Detailed visualization and reporting

## 🏗️ Architecture

The pipeline consists of the following key components:

```
model_compression_pipeline/
├── src/               # Source code
│   ├── data/          # Data loading and preprocessing
│   ├── models/        # Model architectures and training
│   ├── compression/   # Compression techniques implementation
│   ├── evaluation/    # Benchmarking and comparison
│   └── utils/         # Helper utilities
├── experiments/       # Jupyter notebooks for experiments
├── docs/              # Documentation
└── results/           # Saved models and metrics
```

### Compression Techniques

<table>
  <tr>
    <th>Technique</th>
    <th>Description</th>
    <th>Benefits</th>
  </tr>
  <tr>
    <td><b>Pruning</b></td>
    <td>Removes weights based on magnitude, importance, or structure</td>
    <td>Reduces model size and computation with minimal accuracy impact</td>
  </tr>
  <tr>
    <td><b>Quantization</b></td>
    <td>Converts 32-bit floats to lower-precision (8-bit, 4-bit, 2-bit)</td>
    <td>Significantly decreases model size and improves inference speed</td>
  </tr>
  <tr>
    <td><b>Knowledge Distillation</b></td>
    <td>Trains compact models using the output of larger models</td>
    <td>Creates smaller, faster models that retain knowledge from larger ones</td>
  </tr>
  <tr>
    <td><b>Lottery Ticket Hypothesis</b></td>
    <td>Finds sparse subnetworks with comparable performance to full networks</td>
    <td>Identifies highly efficient subnetworks that train effectively from initialization</td>
  </tr>
</table>

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/1Utkarsh1/model-compression-pipeline.git
cd model-compression-pipeline

# Install dependencies
pip install -r requirements.txt
```

## 📊 Examples

### Full Pipeline Workflow

```python
# Train a baseline model and apply all compression techniques
python src/main.py --mode full --model resnet50 --dataset cifar10
```

### Individual Techniques

<details>
<summary><b>Baseline Model Training</b></summary>

```bash
# Train baseline ResNet50 on CIFAR-10
python src/main.py --mode baseline --model resnet50 --dataset cifar10 --epochs 100
```

</details>

<details>
<summary><b>Pruning</b></summary>

```bash
# Apply magnitude-based pruning with 50% sparsity
python src/main.py --mode prune --model resnet50 --dataset cifar10 --prune_rate 0.5 --prune_method magnitude
```

</details>

<details>
<summary><b>Quantization</b></summary>

```bash
# Apply 8-bit post-training quantization
python src/main.py --mode quantize --model resnet50 --dataset cifar10 --bits 8 --quantize_method post_training
```

</details>

<details>
<summary><b>Knowledge Distillation</b></summary>

```bash
# Distill from ResNet50 to ResNet18
python src/main.py --mode distill --model resnet50 --dataset cifar10 --student resnet18
```

</details>

<details>
<summary><b>Lottery Ticket Hypothesis</b></summary>

```bash
# Apply Lottery Ticket pruning with 5 iterations
python src/main.py --mode lottery_ticket --model resnet50 --dataset cifar10 --lottery_iterations 5 --lottery_prune_percent 0.2
```

</details>

<details>
<summary><b>Generate Comparison Report</b></summary>

```bash
# Generate a comprehensive HTML report comparing all techniques
python src/main.py --mode report --model resnet50 --dataset cifar10
```

</details>

## 📈 Results

Here's a comparison of different compression techniques applied to ResNet50 on CIFAR-10:

<div align="center">
<table>
  <tr>
    <th>Model</th>
    <th>Accuracy</th>
    <th>Size (MB)</th>
    <th>Inference Time (ms)</th>
    <th>Memory Usage (MB)</th>
  </tr>
  <tr>
    <td>Baseline (ResNet50)</td>
    <td>92.5%</td>
    <td>97.8</td>
    <td>125</td>
    <td>550</td>
  </tr>
  <tr>
    <td>Pruned (50%)</td>
    <td>91.8%</td>
    <td>49.2</td>
    <td>110</td>
    <td>320</td>
  </tr>
  <tr>
    <td>Quantized (8-bit)</td>
    <td>92.1%</td>
    <td>24.6</td>
    <td>85</td>
    <td>210</td>
  </tr>
  <tr>
    <td>Distilled (ResNet18)</td>
    <td>89.3%</td>
    <td>44.7</td>
    <td>60</td>
    <td>290</td>
  </tr>
  <tr>
    <td>Lottery Ticket (5 iter)</td>
    <td>91.2%</td>
    <td>31.5</td>
    <td>105</td>
    <td>280</td>
  </tr>
</table>
</div>

### Performance Visualizations

<div align="center">
<img src="https://via.placeholder.com/400x300?text=Accuracy+vs+Size" alt="Accuracy vs Size" width="45%">
<img src="https://via.placeholder.com/400x300?text=Accuracy+vs+Latency" alt="Accuracy vs Latency" width="45%">
</div>

## 🔮 Advanced Usage

### Custom Models

You can easily extend the pipeline to support custom models:

```python
# In src/models/custom_model.py
class MyCustomModel(nn.Module):
    def __init__(self, num_classes):
        super(MyCustomModel, self).__init__()
        # Define your model architecture
        self.features = ...
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Then register it in load_baseline_model function
```

### Custom Datasets

Support for new datasets can be added as follows:

```python
# In src/data/data_loader.py
# Add to get_transforms and load_dataset functions
elif dataset_name.lower() == 'my_dataset':
    # Define transforms
    train_transforms = transforms.Compose([...])
    test_transforms = transforms.Compose([...])
    
    # Load dataset
    train_dataset = MyDataset(...)
    val_dataset = MyDataset(...)
    test_dataset = MyDataset(...)
```

### Custom Compression Techniques

The modular architecture allows adding new compression techniques:

```python
# In src/compression/my_technique.py
def apply_my_compression(model, ...):
    # Implement your compression logic
    return compressed_model
```

## 🗺️ Roadmap

- [x] Implement basic pruning techniques
- [x] Implement quantization (8-bit, 4-bit, 2-bit)
- [x] Implement knowledge distillation
- [x] Add Vision Transformer support
- [x] Add Lottery Ticket Hypothesis implementation
- [ ] Support for hardware-aware compression
- [ ] Add NLP model support (BERT, GPT variants)
- [ ] Deploy compressed models to mobile/edge devices
- [ ] Add AutoML for finding optimal compression strategies
- [ ] Support for continuous compression during training

## 📑 Publications

If you use this work in your research, please cite:

```bibtex
@software{model_compression_pipeline,
  author = {Utkarsh Rajput},
  title = {Model Compression Pipeline},
  year = {2025},
  url = {https://github.com/1Utkarsh1/model-compression-pipeline}
}
```

## 👥 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Getting Started with Development

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [PyTorch](https://pytorch.org/) for the deep learning framework
- [TensorFlow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization) for inspiration on compression techniques
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) for transformer model implementations
- [Jonathan Frankle and Michael Carbin](https://arxiv.org/abs/1803.03635) for the Lottery Ticket Hypothesis

---

<div align="center">
    <b>Made with ❤️ by the Model Compression Pipeline Team</b><br>
    <a href="https://github.com/1Utkarsh1">Github</a> •
    <a href="#">Website</a> •
    <a href="#">Contact</a>
</div> 
