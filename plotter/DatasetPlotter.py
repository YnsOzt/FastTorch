import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def plot_classes_distributions(train_dataset, test_dataset, val_dataset=None, figsize=(20, 7)):
    def compute_distribution(dataset):
        distribution = {}
        for (x, y) in tqdm.tqdm(dataset):
            y = dataset.dataset.classes[y] if type(dataset) == torch.utils.data.dataset.Subset else dataset.classes[y]
            if y in distribution:
                distribution[y] += 1
            else:
                distribution[y] = 1
        return distribution

    print("processing training data")
    train_distribution = compute_distribution(train_dataset)
    print("processing test data")
    test_distribution = compute_distribution(test_dataset)
    if val_dataset is not None:
        print("processing validation data")
        val_distribution = compute_distribution(val_dataset)

    fig, axes = plt.subplots(nrows=1, ncols=2 if val_dataset is None else 3, figsize=figsize)

    axes[0].set_title('Training set')
    for item in train_distribution.keys():
        axes[0].bar(item, train_distribution[item])

    axes[1].set_title('Test set')
    for item in test_distribution.keys():
        axes[1].bar(item, test_distribution[item])

    if val_dataset is not None:
        axes[2].set_title('Validation set')
        for item in val_distribution.keys():
            axes[2].bar(item, val_distribution[item])


def plot_images(data_loader, classes, figsize=(15, 15), std_mean=None):
    X, y = next(iter(data_loader))

    #  Show the data then stops
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=nrows_ncols,  # creates 2x2 grid of axes
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