import tqdm
import torch
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np


class Classifier:
    def __init__(self, model, training_opts, train_dataloader, test_dataloader, val_dataloader=None, device='cpu',
                 model_path="best_model.pth"):
        """
        :param model: the model to train
        :param training_opts: the model options, this parameters should be a dict containing those informations:
        {
            epochs : int (number of epochs) [OPT],
            optimizer: optimizer,
            criterion: criterion,
            early_stopping_patience: int (patience of early stopping) [OPT]
        }
        :param train_dataloader: The dataloader of the training set
        :param test_dataloader: The dataloader of the test set
        :param val_dataloader: The dataloader of validation set if not specified, use the test set as the val set [OPT]
        :param device: device to run your model on [ cpu | cuda ] [OPT, DEFAULT=cpu]
        :param model_path: path where the best_model will be saved on [OPT]
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.val_dataloader = val_dataloader
        self.model_path = model_path

        # if the dataset is an instance of subset, we'll reach it by accessing the dataset of subset
        if isinstance(self.train_dataloader.dataset, torch.utils.data.dataset.Subset):
            self.classes = self.train_dataloader.dataset.dataset.classes
        else:
            self.classes = self.train_dataloader.dataset.classes

        # Test set will be the validation set if any validation set isn't specified
        if self.val_dataloader is None:
            self.val_dataloader = test_dataloader

        self.epochs = training_opts['epochs'] if 'epochs' in training_opts else 100
        try:
            self.optimizer = training_opts['optimizer']
        except Exception:
            raise Exception("You should specify an optimize !")

        try:
            self.criterion = training_opts['criterion']
        except Exception:
            raise Exception("You should specify a criterion !")

        self.early_stopping_patience = training_opts[
            'early_stopping_patience'] if 'early_stopping_patience' in training_opts else 5

        if device not in ('cpu', 'cuda'):
            raise Exception('Device should be cpu or cuda')
        self.device = device

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

        self.predictions = torch.tensor([], dtype=torch.long)
        self.gts = torch.tensor([], dtype=torch.long)

        self.VALIDATION = "validation"
        self.TEST = "test"

    def train(self):
        """
        Training loop
        :return:
        """
        self.model.to(self.device)
        print(self.model)
        current_patience = self.early_stopping_patience
        best_accuracy = 0

        for epoch in range(self.epochs):
            print("-" * 80)

            # Forward and backward pass
            print("Epoch : {}/{}".format(epoch + 1, self.epochs))
            print("Starting training")
            current_loss = self._train_epoch()

            # Validation step
            print("Starting validation")
            validation_loss, validation_accuracy = self._validate(self.val_dataloader, self.VALIDATION)

            # Add to lists to plot the training stats after training
            self.train_losses.append(current_loss)
            self.val_losses.append(validation_loss)
            self.val_accuracies.append(validation_accuracy)

            # Early stopping
            if validation_accuracy > best_accuracy:
                print("Accuracy increased from {} to {} ... Saving model as {}".format(best_accuracy,
                                                                                       validation_accuracy, self.model_path))
                best_accuracy = validation_accuracy
                current_patience = self.early_stopping_patience
                self._save_model()
            else:
                current_patience -= 1

            print(
                "Training Loss: {}, Validation Loss: {}, Validation Accuracy: {}, early_stopping_patience: {}".format(
                    current_loss, validation_loss, validation_accuracy, current_patience))
            if current_patience == 0:
                print("Training stopped due to early stopping patience !")
                break

    def test(self):
        """
        Test loop
        :return:
        """
        print("-" * 80)
        print("Starting testing:")
        _, test_accuracy = self._validate(self.test_dataloader, self.TEST)
        print("Accuracy of {} for the test test".format(test_accuracy))

    def _train_epoch(self):
        """
        Train an entier epoch
        :return: the loss of this epoch
        """
        self.model.train()
        current_loss = 0
        num_iter = 0
        for X, y in tqdm.tqdm(self.train_dataloader, position=0, leave=True):
            self.optimizer.zero_grad()
            X = X.to(self.device)
            y = y.to(self.device)

            outputs = self.model(X)

            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            current_loss += loss.item()
            num_iter += 1


        return round(current_loss / len(self.train_dataloader), 4)

    def _validate(self, dataset, mode):
        """
        Test or validation loop
        :param dataset: the dataset to run the test or validation on
        :param mode: which mode we are on [ test / validation ]
        :return: the loss and the accuracy of the validation or test
        """
        current_loss = 0
        current_accuracy = 0
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for X, y in tqdm.tqdm(dataset, position=0, leave=True):
                X = X.to(self.device)
                y = y.to(self.device)
                outputs = self.model(X)
                _, predicted = torch.max(outputs.data, -1)

                # Compute loss
                loss = self.criterion(outputs, y)
                current_loss += loss.item()

                # Compute accuracy
                total += y.size(0)
                correct += (predicted == y).sum().item()

                if mode == self.TEST:
                    self.predictions = torch.cat(
                        (self.predictions, predicted.to('cpu'))
                        , dim=0
                    )
                    self.gts = torch.cat(
                        (self.gts, y.to('cpu'))
                        , dim=0
                    )

            current_loss = round(current_loss / len(dataset), 4)
            current_accuracy = round((100 * correct / total), 4)

        return current_loss, current_accuracy

    def _save_model(self):
        """
        save the model
        :return:
        """
        if ".pth" not in self.model_path:
            self.model_path += ".pth"
        torch.save(self.model.state_dict(), self.model_path)

    def plot_training_stats(self, figsize=(15, 7)):
        """
        plot the training loss, validation loss and the validation accuracy
        :param figsize: figsize (width, height) [OPT]
        :return:
        """
        if len(self.train_losses) > 0:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
            axes[0].plot(self.train_losses, 'r', label='Training loss')
            axes[0].plot(self.val_losses, 'b', label='Validation loss')
            axes[1].scatter([i for i in range(len(self.val_accuracies))], self.val_accuracies, c='green',
                            label='Validation accuracy')
            axes[1].plot([i for i in range(len(self.val_accuracies))], self.val_accuracies)
            axes[1].get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            axes[0].get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            plt.setp(axes[::], xlabel='epoch')
            plt.setp(axes[0], ylabel='Loss')
            plt.setp(axes[1], ylabel='Accuracy')
            plt.show()
        else:
            raise Exception("You should train your model first")

    def plot_random_predictions(self, std_mean=None):
        """
        show predictions on random data
        :param std_mean: (std_dev, mean_var), to show the images withouth the modification applied on [OPT]
        :return:
        """
        self.model.eval()
        with torch.no_grad():
            X, y = next(iter(self.test_dataloader))
            X = X.to(self.device)
            y = y.to(self.device)
            # take juste the 16 first
            X = X[0:16]
            y = y[0:16]
            preds_indices = torch.argmax(self.model(X), -1)  # transforms model predictions into indices
            #  Show the data then stops
            fig = plt.figure(figsize=(16, 16))
            grid = ImageGrid(fig, 111,  # similar to subplot(111)
                             nrows_ncols=(4, 4),  # creates 2x2 grid of axes
                             axes_pad=0.35,  # pad between axes in inch.
                             )

            for ax, im, lb, pred in zip(grid, X, y, preds_indices):
                # Iterating over the grid returns the Axes.
                inp = im.permute(1, 2, 0).detach().cpu().numpy()
                if std_mean is not None:
                    inp = np.array(std_mean[0] * inp + np.array(std_mean[1]))
                # if the image is in grayscale
                if inp.shape[2] == 1:
                    inp = inp.squeeze()
                ax.imshow(inp)
                ax.set_title("{} (GT:{})".format(self.classes[pred], (self.classes[lb])))

            plt.axis("off")
            plt.ioff()
            plt.show()

    # this code is taken from https://deeplizard.com/learn/video/0LhiS6yu2qQ
    def plot_confusion_matrix(self):
        """
        Plot the confusion matrix
        :return:
        """
        if len(self.gts) > 0:
            cm = confusion_matrix(self.gts, self.predictions)
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Confusion matrix")
            plt.colorbar()
            tick_marks = np.arange(len(self.classes))
            plt.xticks(tick_marks, self.classes, rotation=45)
            plt.yticks(tick_marks, self.classes)

            fmt = 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.show()
        else:
            raise Exception("You should train your model first!")


if __name__ == '__main__':
    import torch.nn as nn
    import torch.nn.functional as F
    import torch
    import torch.optim as optim
    import torchvision
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader


    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3)
            self.conv2 = nn.Conv2d(32, 64, 5)
            self.lin = nn.Linear(64 * 22 * 22, len(mnist_trainset.dataset.classes))

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(-1, 64 * 22 * 22)
            x = self.lin(x)
            return F.log_softmax(x, dim=-1)


    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
    mnist_trainset, mnist_valset = torch.utils.data.random_split(mnist_trainset, (55000, 5000))
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms)

    train_loader = DataLoader(mnist_trainset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(mnist_valset, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_testset, batch_size=32, shuffle=True, drop_last=True)

    model = Model()
    training_opts = {
        "epochs": 5,
        "criterion": nn.CrossEntropyLoss(),
        "optimizer": optim.SGD(model.parameters(), lr=0.01),
        "early_stopping_patience": 2
    }

    clf = Classifier(model, training_opts, train_loader, test_loader, val_loader, device="cuda")

    clf.train()
    clf.test()
    clf.plot_training_stats()
    clf.plot_random_predictions()
    clf.plot_confusion_matrix()
