# Documentation Fast_Torch
## Table of content
**[ 1. Classifier](#classifier)**

**[2. DatasetPlotter](#datasetplotter)**

## Classifier
<div id="classifier"/>
<strong>CLASS:</strong> fast_torch.model_wrapper.Classifier(self, model, training_opts, train_dataloader, test_dataloader, val_dataloader=None, device='cpu',
                 model_path="best_model.pth")

### Parameters
* <strong>model</strong>: the model (a class which extends torch.nn.modules) to train
* <strong>training_opts</strong>: he model options, this parameters should be a dict containing those informations<br>
{<br>
<strong>epochs:</strong> int (number of epochs) [OPT], <br>
<strong>optimizer:</strong> the optimizer to use,<br>
<strong>criterion:</strong> the criterion to use,<br>
<strong>early_stopping_patience:</strong> int (patience of early stopping) [OPT]
<br>}
* <strong>train_dataloader:</strong> The dataloader of the training set
* <strong>test_dataloader:</strong> The dataloader of the test set
* <strong>val_dataloader:</strong> The dataloader of validation set, if not specified, it'll use the test set as the val set [OPT]
* <strong>device:  </strong>device to run your model on [ cpu | cuda ] [OPT]
* <strong>model_path:</strong> path where the best_model will be saved on [OPT]

### Methods
* <strong>train():</strong> Starts the training loop
* <strong>test(): </strong> Starts the test loop 
* <strong>plot_training_stats(figsize=(15, 7)):</strong> plot the training loss, validation loss and the validation accuracy
    * <strong>figsize:</strong> figsize (width, height) [OPT]
* <strong>plot_random_predictions(std_mean=None):</strong> show predictions of the model on random data (taken from test set)
    * <strong>std_mean:</strong> (std_dev, mean_var), to show the images withouth the modification applied on [OPT]
* <strong>plot_confusion_matrix():</strong> Plot the confusion matrix

## DatasetPlotter
<div id="datasetplotter"/>
<strong>CLASS:</strong> fast_torch.plotter.DatasetPlotter

All methods are static methods, you don't need to initialize anything to call them.

### Methods
* <strong>plot_classes_distributions(train_dataloader, test_dataloader, val_dataloader=None, figsize=(20, 7)) :</strong> Plot classses distributions for the given datasets
    * <strong>train_dataloader:</strong> The training dataloder
    * <strong>test_dataloader:</strong> The test dataloader
    * <strong>val_dataloader:</strong> The val dataloader [OPT]
    * <strong>figsize: </strong> size of figure (width, height) [OPT]
* <strong>plot_images(data_loader, figsize=(15, 15), std_mean=None): </strong> plot images taken from the given dataloader
    * <strong>data_loader:</strong> the dataloader which contains the images
    * <strong>figsize: </strong> the size of the figure
    * <strong>std_mean:</strong> (std_dev, mean_var), to show the images withouth the modification applied on [OPT]
