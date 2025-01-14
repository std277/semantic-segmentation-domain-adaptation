import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import math

import numpy as np

from datasets import LoveDADatasetLabel, MEAN, STD

CLASS_COLOR = {
    LoveDADatasetLabel.BACKGROUND: (0.0, 0.0, 1.0),     # Background - Blue
    LoveDADatasetLabel.BUILDING: (0.0, 1.0, 0.0),       # Building - Green
    LoveDADatasetLabel.ROAD: (1.0, 0.0, 0.0),           # Road - Red
    LoveDADatasetLabel.WATER: (0.0, 1.0, 1.0),          # Water - Cyan
    LoveDADatasetLabel.BARREN: (1.0, 1.0, 0.0),         # Barren - Yellow
    LoveDADatasetLabel.FOREST: (1.0, 0.0, 1.0),         # Forest - Magenta
    LoveDADatasetLabel.AGRICULTURE: (0.5, 0.5, 0.5),    # Agriculture - Gray
}




def get_mask_color_image(np_image, np_mask):
    np_mask = np.repeat(np_mask[:, :, np.newaxis], 3, axis=2)

    np_mask_color_image = np_image.copy()
    for label in LoveDADatasetLabel:
        np_mask_color_image = np.where(
            np_mask == label.value, CLASS_COLOR[label], np_mask_color_image)

    return np_mask_color_image


def get_boundary_color_image(np_image, np_boundary):
    np_boundary_color_image = np.repeat(np_boundary[:, :, np.newaxis], 3, axis=2)

    return np_boundary_color_image




def plot_dataset_entry(image, mask, boundary, alpha=1., title=None, description=None, show=True):
    fig, axes = plt.subplots(1, 3, figsize=(17, 7))

    axes = axes.flatten()

    if title is not None:
        fig.suptitle(title)

    np_image = image.numpy().transpose((1, 2, 0)) * np.array(STD) + np.array(MEAN)
    axes[0].imshow(np_image)
    axes[0].axis("off")
    axes[0].set_title("Image")

    np_mask = mask.numpy()
    np_mask_color_image = get_mask_color_image(np_image, np_mask)
    axes[1].imshow(np_image)
    axes[1].imshow(np_mask_color_image, alpha=alpha)
    axes[1].axis("off")
    axes[1].set_title("Mask")

    np_boundary = boundary.numpy()
    np_boundary_color_image = get_boundary_color_image(np_image, np_boundary)
    axes[2].imshow(np_image)
    axes[2].imshow(np_boundary_color_image, alpha=alpha)
    axes[2].axis("off")
    axes[2].set_title("Boundary")

    patches = [mpatches.Patch(color=CLASS_COLOR[label], label=label.name.lower().capitalize()) for label in LoveDADatasetLabel]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    if description is not None:
        fig.text(0.5, 0.1, description, ha='center', fontsize=12, wrap=True)

    if show:
        plt.show()






def plot_prediction(image, mask, prediction, alpha=1., title=None, description=None, show=True):
    fig, axes = plt.subplots(1, 3, figsize=(17, 8))

    axes = axes.flatten()

    if title is not None:
        fig.suptitle(title)

    np_image = image.numpy().transpose((1, 2, 0)) * np.array(STD) + np.array(MEAN)
    axes[0].imshow(np_image)
    axes[0].axis("off")
    axes[0].set_title("Image")

    np_mask = mask.numpy()
    np_mask_color_image = get_mask_color_image(np_image, np_mask)
    axes[1].imshow(np_image)
    axes[1].imshow(np_mask_color_image, alpha=alpha)
    axes[1].axis("off")
    axes[1].set_title("Mask")

    np_prediction = prediction.numpy()
    np_prediction_color_image = get_mask_color_image(np_image, np_prediction)
    axes[2].imshow(np_image)
    axes[2].imshow(np_prediction_color_image, alpha=alpha)
    axes[2].axis("off")
    axes[2].set_title("Prediction")

    patches = [mpatches.Patch(color=CLASS_COLOR[label], label=label.name.lower().capitalize()) for label in LoveDADatasetLabel]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    if description is not None:
        fig.text(0.5, 0.02, description, ha='center', fontsize=12, wrap=True)

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
            np_image = image.numpy().transpose((1, 2, 0)) * np.array(STD) + np.array(MEAN)
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
            np_image = image.numpy().transpose((1, 2, 0)) * np.array(STD) + np.array(MEAN)
            axes[i].imshow(np_image)

    if show:
        plt.show()


def inspect_dataset(trainloader, valloader):
    for type, loader in zip(("Train", "Val"), (trainloader, valloader)):
        if loader is not None:
            it = iter(loader)
            images, masks, boundaries = next(it)

            for image, mask, boundary in zip(images, masks, boundaries):
                plot_dataset_entry(image, mask if type != "Test" else None, boundary, alpha=0.4, title=f"{type} image sample")

            # plot_batch(images, masks if type!="Test" else None, alpha=0.3, title=f"{type} image batch")





def plot_loss(train_losses, val_losses, model_number, res_dir):
    fig = plt.figure()
    plt.title("Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.savefig(f"{res_dir}/plots/loss_{model_number}.pdf")
    plt.close(fig)


def plot_mIoU(train_mIoUs, val_mIoUs, model_number, res_dir):
    fig = plt.figure()
    plt.title("Mean Intersection over Union")
    plt.ylabel("mIoU")
    plt.xlabel("Epoch")
    plt.plot(train_mIoUs, label="Train mIoU")
    plt.plot(val_mIoUs, label="Val mIoU")
    plt.legend()
    plt.savefig(f"{res_dir}/plots/mIoU_{model_number}.pdf")
    plt.close(fig)


def plot_learning_rate(learning_rates, model_number, res_dir):
    fig = plt.figure()
    plt.title("Learning rate")
    plt.ylabel("learning rate")
    plt.xlabel("Epoch")
    plt.plot(learning_rates, label="Learning Rate")
    plt.legend()
    plt.savefig(f"{res_dir}/plots/learning_rate_{model_number}.pdf")
    plt.close(fig)

