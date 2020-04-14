import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def plot_classes_distributions(train_dataloader, test_dataloader, val_dataloader=None, figsize=(20, 7)):
    """
    Plot classses distributions for the given datasets
    :param train_dataloader: The training dataloder
    :param test_dataloader: The test dataloader
    :param val_dataloader: The val dataloader[OPT]
    :param figsize: size of figure (width, height) [OPT, DEFAULT = (20, 7)]
    :return:
    """

    def compute_distribution(dataset):
        distribution = {}
        for (x, y) in tqdm.tqdm(dataset):
            for lb in y:
                current_y = dataset.dataset.dataset.classes[lb] if isinstance(dataset.dataset,
                                                                              torch.utils.data.dataset.Subset) else \
                    dataset.dataset.classes[lb]
                if current_y in distribution:
                    distribution[current_y] += 1
                else:
                    distribution[current_y] = 1
        return distribution

    print("processing training data")
    train_distribution = compute_distribution(train_dataloader)
    print("processing test data")
    test_distribution = compute_distribution(test_dataloader)
    if val_dataloader is not None:
        print("processing validation data")
        val_distribution = compute_distribution(val_dataloader)

    fig, axes = plt.subplots(nrows=1, ncols=2 if val_dataloader is None else 3, figsize=figsize)

    axes[0].set_title('Training set')
    for item in train_distribution.keys():
        axes[0].bar(item, train_distribution[item])

    axes[0].tick_params(axis='x', labelrotation=45)

    axes[1].set_title('Test set')
    for item in train_distribution.keys():
        axes[1].bar(item, test_distribution[item])

    axes[1].tick_params(axis='x', labelrotation=45)

    if val_dataloader is not None:
        axes[2].set_title('Validation set')
        for item in train_distribution.keys():
            axes[2].bar(item, val_distribution[item])

    axes[2].tick_params(axis='x', labelrotation=45)

    plt.show()


def plot_images(data_loader, figsize=(15, 15), std_mean=None):
    """
    plot images from the given dataloader
    :param data_loader: the dataloader which contains the images
    :param figsize: the size of the figure
    :param std_mean: (std_dev, mean_var), to show the images withouth the modification applied on [OPT]
    :return:
    """
    classes = data_loader.dataset.dataset.classes if isinstance(data_loader.dataset,
                                                                torch.utils.data.dataset.Subset) else \
        data_loader.dataset.classes
    X, y = next(iter(data_loader))

    #  Show the data then stops
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(4, 4),  # creates 4x4 grid of axes
                     axes_pad=0.35,  # pad between axes in inch.
                     )

    for ax, im, lb in zip(grid, X, y):
        # Iterating over the grid returns the Axes.
        inp = im.permute(1, 2, 0).detach().cpu().numpy()
        if std_mean is not None:
            inp = np.array(std_mean[0] * inp + np.array(std_mean[1]))
        if inp.shape[2] == 1:
            inp = inp.squeeze()
        ax.imshow(inp)
        ax.set_title(classes[lb])
    plt.axis("off")
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    import torch
    import torch.optim as optim
    import torchvision
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader
    import tqdm

    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
    mnist_trainset, mnist_valset = torch.utils.data.random_split(mnist_trainset, (55000, 5000))
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms)

    train_loader = DataLoader(mnist_trainset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(mnist_valset, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_testset, batch_size=32, shuffle=True, drop_last=True)

    plot_classes_distributions(train_loader, test_loader, val_loader)
    plot_images(train_loader)
