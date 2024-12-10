# MNIST CNN Training Visualization

This project implements a 4-layer CNN for MNIST digit classification with real-time training visualization.

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy

Install requirements: 
```
pip install -r requirements.txt
```

## Project Structure
.
├── mnist_cnn.py # Main CNN model and training code
├── server.py # Simple HTTP server
├── index.html # Visualization webpage
└── results/ # Directory for training logs and result

## How to Run

1. Make sure all required packages are installed
2. Start the visualization server:
   ```bash
   python server.py
   ```
3. In a new terminal, start the training:
   ```bash
   python mnist_cnn.py
   ```
4. Open your web browser and go to http://localhost:8000 (should open automatically)
5. Watch the training progress in real-time
6. After training completes, you'll see the results of 10 random test samples

## Model Architecture

The CNN consists of:
- 4 convolutional layers with increasing filters (16, 32, 64, 128)
- ReLU activation functions
- MaxPooling layers
- 2 fully connected layers
- Output layer with 10 classes (digits 0-9)

## Training Details

- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 64
- Epochs: 10
- Dataset: MNIST (60,000 training images, 10,000 test images)

## Visualization

The visualization is done using a simple HTML page that displays the training progress and results.

