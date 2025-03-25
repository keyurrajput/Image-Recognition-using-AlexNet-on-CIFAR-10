# AlexNet for CIFAR-10 Image Classification

## Overview
This project implements the AlexNet convolutional neural network architecture for image classification on the CIFAR-10 dataset. AlexNet, designed by Alex Krizhevsky and supervised by Geoffrey Hinton, was a groundbreaking architecture that won the 2012 ImageNet competition and revolutionized the field of computer vision with deep learning approaches.

## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images across 10 different classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset is divided into 50,000 training images and 10,000 test images. In this implementation, we further split the training data into 45,000 training samples and 5,000 validation samples.

## Model Architecture
The implementation adapts the original AlexNet architecture to work with the smaller 32x32 CIFAR-10 images (compared to ImageNet's 224x224). The model consists of:

1. **Convolutional Layers**: Five convolutional layers with ReLU activations
2. **Pooling Layers**: Three max pooling layers to reduce spatial dimensions
3. **Fully Connected Layers**: Three fully connected layers for classification
4. **Regularization**: Dropout layers with 0.5 probability to prevent overfitting

The model was initialized using Kaiming initialization for convolutional layers and Xavier initialization for linear layers to improve training stability and convergence.

## Data Augmentation
To improve model generalization and prevent overfitting, several data augmentation techniques were applied:
- Random rotation (±5 degrees)
- Random horizontal flipping (50% probability)
- Random cropping with padding

## Training Details
- **Optimizer**: Adam with a learning rate of 0.001
- **Loss Function**: Cross-Entropy Loss
- **Batch Size**: 256
- **Epochs**: 100
- **Best Model Selection**: Model with lowest validation loss is saved

## Results
The model achieves approximately 79.38% accuracy on the test set, which is competitive for a classic CNN architecture on the CIFAR-10 dataset. Training accuracy reached 91.82% after 100 epochs, suggesting some overfitting despite the regularization techniques employed.

## Files Description
- `alexnet1.ipynb`: Jupyter notebook containing the complete implementation, training, and evaluation code
- `best-model.pt`: PyTorch saved model state after training
- `cifar10_alexnet_model.pth`: Final model saved in PyTorch format

## Usage
To use this model for inference:

```python
import torch
import torchvision.transforms as transforms
from model import AlexNet  # Assuming model definition is in model.py

# Load the trained model
model = AlexNet(output_dim=10)
model.load_state_dict(torch.load('cifar10_alexnet_model.pth'))
model.eval()

# Prepare image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091], 
                         std=[0.24703223, 0.24348513, 0.26158784])
])

# Prediction function
def predict(image):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output, _ = model(image_tensor)
        prediction = output.argmax(1).item()
    return prediction
```

## Requirements
- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-learn

## Future Improvements
- Implement learning rate scheduling for better convergence
- Explore modern architectures like ResNet or EfficientNet for comparison
- Apply more sophisticated data augmentation techniques
- Experiment with different optimizers and hyperparameters
