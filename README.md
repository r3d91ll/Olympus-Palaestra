# Palaestra

Palaestra is an advanced training framework for large language models, leveraging DeepSpeed for efficient multi-GPU training. It provides a flexible and modular approach to preprocessing data, training models, and evaluating results.

## Project Structure

```
palaestra/
├── askesis/
│   └── deepspeed_regime.py
├── analyzer/
│   ├── analyzer.py
│   └── utils.py
├── config/
│   └── __init__.py
├── data/
│   └── dataset.py
├── evaluation/
│   └── evaluator.py
├── models/
│   └── model_loader.py
├── propone/
│   ├── propone.py
│   └── utils.py
├── training/
│   └── trainer.py
├── utils/
│   ├── accelerate/
│   ├── deepspeed/
│   │   ├── ds_config.json
│   │   └── utils.py
│   └── huggingface/
│       └── huggingface_utils.py
├── old/
│   ├── palaestra.py
│   ├── preprocess_palaestra.py
│   └── model_dataset_analyzer.py
└── main.py
```

## Features

- Modular architecture for easy extension and customization
- Integration with DeepSpeed for efficient large-scale model training
- Automatic model and dataset management using Hugging Face Hub
- Flexible data preprocessing pipeline
- Comprehensive logging and evaluation tools

## Prerequisites

- Docker
- NVIDIA GPU with CUDA support (for GPU acceleration)
- Hugging Face account (for model and dataset management)

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/palaestra.git
   cd palaestra
   ```

2. Configure your Hugging Face credentials:
   - Create a file named `.env` in the project root
   - Add your Hugging Face token:
     ```
     HF_TOKEN=your_huggingface_token_here
     ```

3. Build the Docker images:
   ```
   docker-compose build
   ```

## Configuration

1. Edit the `config/config.yaml` file to set up your training parameters, model selection, and data paths.
2. Adjust the DeepSpeed configuration in `utils/deepspeed/ds_config.json` if needed.

## Usage

1. Start the training process:
   ```
   docker-compose up palaestra
   ```

2. Monitor training progress with TensorBoard:
   ```
   docker-compose up tensorboard
   ```
   Then open `http://localhost:8006` in your web browser.

## Components

- **Propone**: Data preprocessing module (`propone/propone.py`)
- **Analyzer**: Model and dataset compatibility analysis (`analyzer/analyzer.py`)
- **Training**: DeepSpeed-based training loop (`training/trainer.py`)
- **Evaluation**: Model evaluation utilities (`evaluation/evaluator.py`)
- **Utils**: Utility functions for DeepSpeed, Hugging Face, and more

## Extending Palaestra

To add support for new training regimes or optimization techniques:

1. Create a new module in the appropriate directory (e.g., `utils/accelerate/` for Accelerate support)
2. Implement the necessary functions and classes
3. Update `main.py` to incorporate the new functionality

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.