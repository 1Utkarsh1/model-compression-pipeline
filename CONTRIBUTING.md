# Contributing to Model Compression Pipeline

Thank you for your interest in contributing to the Model Compression Pipeline! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We strive to maintain a positive and inclusive environment.

## Ways to Contribute

There are many ways to contribute to this project:

1. Reporting bugs
2. Suggesting new features or enhancements
3. Improving documentation
4. Adding tests
5. Submitting pull requests with code improvements

## Getting Started

### Setting Up the Development Environment

1. Fork the repository
2. Clone your fork to your local machine:
   ```bash
   git clone https://github.com/yourusername/model-compression-pipeline.git
   cd model-compression-pipeline
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bugfix-name
   ```
2. Make your changes
3. Run tests to ensure your changes don't break existing functionality
4. Commit your changes with a clear message
5. Push your branch to your fork
6. Create a pull request to the main repository

## Coding Guidelines

### Code Style

- Follow PEP 8 guidelines for Python code
- Use clear, descriptive variable and function names
- Add docstrings to all functions, classes, and modules
- Include type hints for function parameters and return values

### Testing

- Add tests for new functionality
- Ensure all tests pass before submitting a pull request
- Aim for high code coverage

### Documentation

- Update documentation for any modified functionality
- Document new features thoroughly
- Include examples where appropriate

## Adding New Compression Techniques

When adding a new compression technique to the pipeline:

1. Create a new module in the appropriate directory
2. Implement the technique with a clear interface that integrates with the existing pipeline
3. Add comprehensive documentation
4. Include tests for the new technique
5. Add an example notebook demonstrating the technique
6. Update the README to mention the new technique

## Submitting Pull Requests

When you submit a pull request:

1. Provide a clear description of the changes
2. Reference any related issues
3. Include screenshots or examples if relevant
4. Ensure CI checks pass
5. Request a review from a maintainer

## Questions?

If you have any questions about contributing, please open an issue or reach out to the maintainers.

Thank you for contributing to make the Model Compression Pipeline better! 