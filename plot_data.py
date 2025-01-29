import os
import numpy as np
import matplotlib.pyplot as plt

import pickle

def plot(X, Y0, Y1=None, title="Plot", xlabel="X", ylabel="Y", label0="False", label1="True", save_disk=False, output_dir="./outputs", output_name=None):
    if Y1 and len(Y0) != len(Y1):
        raise ValueError("Non consistent sizes in Y0 and Y1")
    
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    plt.figure(figsize=(10, 6))
    #plt.xscale('log', base=10)
    plt.xticks(rotation=45)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)


    if isinstance(Y0, list) and all(not isinstance(x, list) for x in Y0):           # Y contain single lists
        plt.plot(X, Y0, marker='.', label=label0, color='blue', linewidth=1.6)
        if Y1:
            plt.plot(X, Y1, marker='.', label=label1, color='orange', linewidth=1.6)
        plt.legend(fontsize=12)
    else:                                                                           # Y contain many lists to be compared
        colors = plt.cm.plasma(np.linspace(0, 1, len(Y0)))
        for i in range(len(Y0)):
            plt.plot(X, Y0[i], marker='.', label=label0[i], color=colors[i], linewidth=1.6)
            if Y1:
                plt.plot(X, Y1[i], marker='.', label=label1[i], color=colors[i], linestyle='--', linewidth=1.3)
        plt.legend(fontsize=8)

    plt.tight_layout()
    
    if save_disk:
        os.makedirs(output_dir, exist_ok=True)
        # plt.savefig(f"{output_dir}/{output_name}.pdf")
        plt.savefig(f"{output_dir}/{output_name}.jpg")
    plt.show()

def plot_bar_chart(X, Y0, Y1, title, xlabel, ylabel, label0, label1, save_disk=False, output_dir="./outputs", output_name=None):
    # Configurazione
    x = np.arange(len(X))  # Posizioni delle barre
    width = 0.35  # Larghezza delle barre

    # Creazione del grafico
    fig, ax = plt.subplots(figsize=(10, 6))
    bar1 = ax.bar(x - width/2, Y0, width, label=label0, color='blue')           # 'steelblue'
    bar2 = ax.bar(x + width/2, Y1, width, label=label1, color='orange')         # 'salmon'

    # Etichette e titolo
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(X, rotation=45, ha='right', fontsize=10)
    ax.legend()

    # Griglia
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Mostrare il grafico
    plt.tight_layout()


    plt.axhline(y=25.44, color='red', linestyle=':', linewidth=2., label="Lower bound Rural")
    plt.axhline(y=33.16, color='green', linestyle=':', linewidth=2., label="Upper bound Rural")

    plt.legend()
   
    if save_disk:
        os.makedirs(output_dir, exist_ok=True)
        # plt.savefig(f"{output_dir}/{output_name}.pdf")
        plt.savefig(f"{output_dir}/{output_name}.jpg")

    plt.show()


if __name__ == "__main__":

    os.makedirs("plots", exist_ok=True)
    
    # mIoU bar plot
    
    # X = ["No Data Aug", "HF, SSR", "HF, SSR, GD", "HF, SSR, RC", "HF, SSR, CD", "HF, RC, CD", "HF, SSR, RC, CJ", "HF, SSR, RC, GB"]
    # mIoU_Urban = [39.24, 40.38, 38.04, 38.94, 41.11, 38.14, 37.65, 38.53]
    # mIoU_Rural = [25.44, 27.50, 27.00, 28.30, 27.21, 25.19, 24.30, 26.61]
    # plot_bar_chart(X, mIoU_Urban, mIoU_Rural, "Domain mIoU (%) comparison on different settings", "Data Augmentation Configurations", "mIoU (%)", "Urban", "Rural", True, "./plots", "data_aug_PIDNet_plot_chart")
    


    VERSION = 8

    # GCW iteration weights plot

    with open(f"res/PIDNet_S_DACS_GCW_LDQ_{VERSION}/data/class_weights_iter_history.pkl", "rb") as f:
        class_weights_iter_history = pickle.load(f)

    class_weights_iter_history = np.array(class_weights_iter_history).T

    X = [i+1 for i in range(class_weights_iter_history.shape[1])]

    labels = ["BACKGROUND", "BUILDING", "ROAD", "WATER", "BARREN", "FOREST", "AGRICULTURE"]


    plot(X, class_weights_iter_history, title="Gradual Class Weights comparison on different classes", xlabel="Epoch", ylabel="GCW", label0=labels, save_disk=True, output_dir="./plots", output_name="GCW_iter_hist")




    # GCW epoch weights plot

    with open(f"res/PIDNet_S_DACS_GCW_LDQ_{VERSION}/data/class_weights_epoch_history.pkl", "rb") as f:
        class_weights_epoch_history = pickle.load(f)

    class_weights_epoch_history = np.array(class_weights_epoch_history).T

    X = [i+1 for i in range(class_weights_epoch_history.shape[1])]

    labels = ["BACKGROUND", "BUILDING", "ROAD", "WATER", "BARREN", "FOREST", "AGRICULTURE"]


    plot(X, class_weights_epoch_history, title="Gradual Class Weights comparison on different classes", xlabel="Epoch", ylabel="GCW", label0=labels, save_disk=True, output_dir="./plots", output_name="GCW_epoch_hist")
