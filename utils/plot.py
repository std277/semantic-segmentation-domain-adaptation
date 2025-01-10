import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import math

import numpy as np

from datasets import LoveDADatasetLabel

CLASS_COLOR = {
    LoveDADatasetLabel.BACKGROUND: (0.0, 0.0, 1.0),     # Background - Blue
    LoveDADatasetLabel.BUILDING: (0.0, 1.0, 0.0),       # Building - Green
    LoveDADatasetLabel.ROAD: (1.0, 0.0, 0.0),           # Road - Red
    LoveDADatasetLabel.WATER: (0.0, 1.0, 1.0),          # Water - Cyan
    LoveDADatasetLabel.BARREN: (1.0, 1.0, 0.0),         # Barren - Yellow
    LoveDADatasetLabel.FOREST: (1.0, 0.0, 1.0),         # Forest - Magenta
    LoveDADatasetLabel.AGRICULTURE: (0.5, 0.5, 0.5),    # Agriculture - Gray
}

MEAN = np.array((0.485, 0.456, 0.406))
STD = np.array((0.229, 0.224, 0.225))


def get_mask_color_image(np_image, np_mask):
    np_mask = np.repeat(np_mask[:, :, np.newaxis], 3, axis=2)

    np_mask_color_image = np_image.copy()
    for label in LoveDADatasetLabel:
        np_mask_color_image = np.where(
            np_mask == label.value, CLASS_COLOR[label], np_mask_color_image)

    return np_mask_color_image


def plot_image(image, mask=None, alpha=1., title=None, show=True):
    if alpha == 1 and mask is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        axes = axes.flatten()

        if title is not None:
            fig.suptitle(title)

        np_image = image.numpy().transpose((1, 2, 0)) * STD + MEAN
        axes[0].imshow(np_image)
        axes[0].axis("off")
        axes[0].set_title("Original Image")

        np_mask = mask.numpy()
        np_mask_color_image = get_mask_color_image(np_image, np_mask)
        axes[1].imshow(np_image)
        axes[1].imshow(np_mask_color_image, alpha=1.0)
        axes[1].axis("off")
        axes[1].set_title("Mask Overlay")

        patches = [mpatches.Patch(color=CLASS_COLOR[label], label=label.name.lower().capitalize()) for label in LoveDADatasetLabel]
        plt.legend(handles=patches)

        if show:
            plt.show()
    else:
        plt.figure(figsize=(12, 7))

        if title is not None:
            plt.title(title)

        np_image = image.numpy().transpose((1, 2, 0)) * STD + MEAN
        plt.imshow(np_image)

        if mask is not None:
            np_mask = mask.numpy()
            np_mask_color_image = get_mask_color_image(np_image, np_mask)
            plt.imshow(np_mask_color_image, alpha=alpha)

            patches = [mpatches.Patch(color=CLASS_COLOR[label], label=label.name.lower().capitalize()) for label in LoveDADatasetLabel]
            plt.legend(handles=patches, bbox_to_anchor=(
                1.05, 1), loc='upper left', borderaxespad=0.)

        plt.axis("off")

        if show:
            plt.show()


def plot_batch(images, masks=None, alpha=0.3, title=None, show=True):
    n = len(images)

    fig, axes = plt.subplots(math.ceil(n/4), 4, figsize=(16, 7))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    axes = axes.flatten()

    if title is not None:
        fig.suptitle(title)

    if masks is not None:
        for i, (image, mask) in enumerate(zip(images, masks)):
            axes[i].axis("off")
            np_image = image.numpy().transpose((1, 2, 0)) * STD + MEAN
            np_mask = mask.numpy()
            np_mask_color_image = get_mask_color_image(np_image, np_mask)
            axes[i].imshow(np_image)
            axes[i].imshow(np_mask_color_image, alpha=alpha)
        patches = [mpatches.Patch(color=CLASS_COLOR[label], label=label.name.lower(
        ).capitalize()) for label in LoveDADatasetLabel]
        plt.legend(handles=patches, bbox_to_anchor=(
            1.05, 1), loc='upper left', borderaxespad=0.)
    else:
        for i, image in enumerate(images):
            axes[i].axis("off")
            np_image = image.numpy().transpose((1, 2, 0)) * STD + MEAN
            axes[i].imshow(np_image)

    if show:
        plt.show()


def inspect_dataset(trainloader, valloader, testloader):
    for type, loader in zip(("Train", "Val", "Test"), (trainloader, valloader, testloader)):
        if loader is not None:
            it = iter(loader)
            images, masks = next(it)

            # for image, mask in zip(images, masks):
            #     plot_image(image, mask if type != "Test" else None, alpha=1., title=f"{type} image sample", show=False)

            plot_batch(images, masks if type!="Test" else None, alpha=0.3, title=f"{type} image batch", show=False)

            plt.show()

def inspect_dataset_masks(trainloader, valloader, testloader):
    for type, loader in zip(("Train", "Val", "Test"), (trainloader, valloader, testloader)):
        for images, masks in iter(loader):
            for image, mask in zip(images, masks):
                if 255 in mask:
                    print(f"Mask:\n{mask}")
                    plot_image(image, mask if type != "Test" else None, alpha=1., title=f"{type} image sample with 255 values in mask", show=False)
                    plt.show()



# def plot_training_metrics(
#     train_losses, val_losses, train_accuracies, val_accuracies, learning_rates, model_number, base_dir
# ):
#     fig = plt.figure()
#     plt.title("Loss")
#     plt.ylabel("Loss")
#     plt.xlabel("Epoch")
#     plt.plot(train_losses, label="Train Loss")
#     plt.plot(val_losses, label="Val Loss")
#     plt.legend()
#     plt.savefig(f"{base_dir}/plots/loss_{model_number}.pdf")
#     plt.close(fig)

#     fig = plt.figure()
#     plt.title("Accuracy")
#     plt.ylabel("Accuracy")
#     plt.xlabel("Epoch")
#     plt.plot(train_accuracies, label="Train Accuracy")
#     plt.plot(val_accuracies, label="Val Accuracy")
#     plt.legend()
#     plt.savefig(f"{base_dir}/plots/accuracy_{model_number}.pdf")
#     plt.close(fig)

#     fig = plt.figure()
#     plt.title("Learning rate")
#     plt.ylabel("learning rate")
#     plt.xlabel("Epoch")
#     plt.plot(learning_rates, label="Learning Rate")
#     plt.legend()
#     plt.savefig(f"{base_dir}/plots/learning_rate_{model_number}.pdf")
#     plt.close(fig)

def plot_loss(train_losses, res_dir):
    fig = plt.figure()
    plt.title("Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.plot(train_losses, label="Train Loss")
    # plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.savefig(f"{res_dir}/plots/loss.pdf")
    plt.close(fig)

def plot_learning_rate(learning_rates, res_dir):
    fig = plt.figure()
    plt.title("Learning rate")
    plt.ylabel("learning rate")
    plt.xlabel("Epoch")
    plt.plot(learning_rates, label="Learning Rate")
    plt.legend()
    plt.savefig(f"{res_dir}/plots/learning_rate.pdf")
    plt.close(fig)

