# MNIST CNN Visualization

A real-time visualization system for training a Convolutional Neural Network (CNN) on the MNIST dataset. The project includes a web interface that displays training progress and results in real-time.

## Features

- 4-layer Convolutional Neural Network
- Real-time training visualization with loss curves
- Web-based interface using a simple Python server
- Display of model predictions on random test samples
- Automatic model checkpointing

## Prerequisites

- Python 3.7+
- PyTorch
- torchvision
- numpy
- Web browser with JavaScript enabled

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/mnist-cnn-visualization.git
cd mnist-cnn-visualization
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install torch torchvision numpy
```

## Project Structure

```
.
├── README.md           # Project documentation
├── HowTo.md           # Detailed usage instructions
├── mnist_cnn.py       # CNN model and training code
├── server.py          # Simple HTTP server
├── index.html         # Visualization webpage
└── results/           # Directory for training logs and results
```

## Quick Start

1. Start the visualization server:
```bash
python server.py
```

2. In a new terminal, start the training:
```bash
python mnist_cnn.py
```

3. The web interface will automatically open in your default browser at http://localhost:8000

## Model Architecture

The CNN architecture consists of:
- Input layer (28x28 grayscale images)
- 4 convolutional layers (16→32→64→128 filters)
- ReLU activations and MaxPooling
- 2 fully connected layers
- Output layer (10 classes)

## Visualization Features

- Real-time loss curve plotting
- Batch-wise training progress
- Display of model predictions on test samples
- Automatic updates every 5 seconds

## Results

After training completes:
- Model weights are saved to `results/mnist_cnn.pth`
- Test predictions are saved to `results/test_samples.json`
- Training logs are saved to `results/training_log.json`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- MNIST Dataset: [LeCun et al.](http://yann.lecun.com/exdb/mnist/)
- Plotly.js for visualization
- PyTorch team for the deep learning framework 