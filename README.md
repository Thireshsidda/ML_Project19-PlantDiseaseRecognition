# ML_Project19-PlantDiseaseRecognition

### Plant Disease Recognition with Deep Learning
This project explores using deep learning for plant disease recognition based on leaf images. It utilizes the "New Plant Diseases Dataset" from Kaggle.

### Dataset Exploration

The code verifies the existence of the training, validation, and test directories within the downloaded dataset.

It then iterates over the training directory, listing the subdirectories which represent the different plant disease categories.

### Data Preprocessing and Augmentation

The code defines transformations for the training and validation datasets using torchvision.transforms.

Images are resized to 128x128 pixels.

Random horizontal flips are applied for data augmentation.

Images are converted to tensors and normalized using standard values for image normalization in PyTorch.

### Data Loading

ImageFolder is used to create datasets from the training and validation directories along with their corresponding transformations.

Dataloaders are created for training, validation, and testing using DataLoader. These loaders manage efficient data loading for training and evaluation.

### Visualizing the Data

A function show_batch is defined to display a sample batch of images from the training dataloader.

This helps visualize the data and verify the preprocessing steps.

### Jupyter Notebook Integration (Optional)

The code includes installation and usage of the jovian library, which simplifies notebook management on the Jovian platform (commented out in this example).

### Device Agnostic Training

The code defines functions to detect the available device (CPU or GPU) and transfer data and models to the appropriate device.

This enables training on either CPU or GPU depending on the available hardware.


### Model Architectures

Three different model architectures are implemented:
```
Simple Model: A basic linear model with a single fully-connected layer.
NN Model: A feed-forward neural network with two hidden layers with ReLU activation.
CNN Model: A convolutional neural network with Batch Normalization layers for better convergence.
```

### Model Training and Evaluation

A custom ImageClass class is defined as the base class for all models. It provides training and validation steps with logging and metric calculation functionalities.

Each model inherits from ImageClass and implements its specific forward pass logic.

The fit function trains a model for a specified number of epochs using Adam optimizer with a learning rate scheduler. It also supports weight decay and gradient clipping for better training stability.

The evaluate function calculates the validation loss and accuracy on the validation dataset.


### Next Steps

Train the implemented models and compare their performance.

Explore more advanced CNN architectures like ResNet or EfficientNet.

Experiment with different hyperparameters like learning rate, batch size, and optimizer settings.

Consider using techniques like transfer learning with pre-trained models.

Evaluate the model performance on the test dataset.
