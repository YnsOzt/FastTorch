import tqdm
import torch


class Classifier:
    def __init__(self, model, training_opts, train_dataloader, test_dataloader, val_dataloader=None, device='cpu'):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.val_dataloader = val_dataloader

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
        model.to(self.device)
        print(self.model)
        current_patience = self.early_stopping_patience
        best_accuracy = -1

        for epoch in range(self.epochs):
            print("-" * 80)

            # Forward and backward pass
            print("Epoch : {}/{}".format(epoch + 1, self.epochs))
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
                best_accuracy = validation_accuracy
                current_patience = self.early_stopping_patience
            else:
                current_patience -= 1

            print(
                "Training Loss: {}, Validation Loss: {}, Validation Accuracy: {}, early_stopping_patience: {}".format(
                    current_loss, validation_loss, validation_accuracy, current_patience))
            if current_patience == 0:
                print("Training stopped due to early stopping patience !")
                break

    def test(self):
        print("-"*80)
        print("Starting testing:")
        _, test_accuracy = self._validate(self.test_dataloader, self.TEST)
        print("Accuracy of {} for the test test".format(test_accuracy))

    def _train_epoch(self):
        self.model.train()
        current_loss = 0
        for X, y in tqdm.tqdm(self.train_dataloader):
            self.optimizer.zero_grad()
            X = X.to(self.device)
            y = y.to(self.device)

            outputs = self.model(X)

            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            current_loss += loss.item()

        return round(current_loss / len(self.train_dataloader), 4)

    def _validate(self, dataset, mode):
        current_loss = 0
        current_accuracy = 0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for X, y in tqdm.tqdm(dataset):
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

    def plot_stats(self):
        pass


if __name__ == '__main__':
    import torch.nn as nn
    import torch.nn.functional as F


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


    import torch
    import torch.optim as optim
    import torchvision
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader

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
        "epochs": 1,
        "criterion": nn.CrossEntropyLoss(),
        "optimizer": optim.SGD(model.parameters(), lr=0.01),
        "early_stopping_patience": 5
    }
    clf = Classifier(model, training_opts, train_loader, test_loader, val_loader, device="cuda")

    clf.train()
    clf.test()
