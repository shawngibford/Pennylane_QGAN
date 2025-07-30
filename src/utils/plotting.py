import matplotlib.pyplot as plt
import numpy as np

def plot_data(real_data, fake_data, epoch):
    """
    Plots real vs. generated data.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(real_data, bins=50, density=True, alpha=0.7, label='Real Data')
    plt.hist(fake_data, bins=50, density=True, alpha=0.7, label='Generated Data')
    plt.title(f"Data Distribution at Epoch {epoch}")
    plt.legend()
    plt.savefig(f'data_dist_epoch_{epoch}.png')
    plt.close() 