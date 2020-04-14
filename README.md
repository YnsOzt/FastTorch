# FastTorch

Library that implements boiler plate code in PyTorch for training, testing and plotting your model

## Installation
run the following command then you are ready to go ! 
```
pip install fast_torch
```

## Usage
Take a look at the complete [Documentation](./documentation/README.md) of this framework.

A complete example is available on the [Notebook](./examples/FAST_TORCH_MNIST_EXAMPLE.ipynb).
### Plot stats about your datasets
```python
# Import the plotting module of the library
import fast_torch.plotter as ftplot

# Will plot some random images of the dataloader that you've passed in the parameter
ftplot.plot_images(train_dataloader)

# Will plot the class distributions of your train, val, test datasets
ftplot.plot_classes_distributions(train_dataloader, test_dataloader, val_dataloader)
```
### Train and test your model 
```python
# Import the model_wrapper module
import fast_torch.model_wrapper as mw


# Initialize your model
model = Model()

# Initialize the options
training_opts = {
    "epochs": 5,
    "criterion": nn.CrossEntropyLoss(),
    "optimizer": optim.SGD(model.parameters(), lr=0.01),
    "early_stopping_patience": 2
}

# Instantiate the classifier wrapper 
clf = mw.Classifier(model, training_opts, train_loader, test_loader, val_loader, device="cuda")

# Train your model
clf.train()
# Test your model
clf.test()

# Plot the training stats (Training loss, Vaildation loss + accuracy)
clf.plot_training_stats()
# Plot random prediction made by your trained model
clf.plot_random_predictions()
# Plot the confusion matrix
clf.plot_confusion_matrix()
```

## TODOS
* Clean the code
* Make the confusion matrix figsize flexible
* Add Learning rate decay to the 'Classifier'
* Add other model wrappers
* Make the plot functions more flexible (working with other type of dataset, not only dataloader)
