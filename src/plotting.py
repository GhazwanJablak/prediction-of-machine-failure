import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

def group_barplot(
        column_name:str, 
        data:Dict
        ):
    """
    Bar plot of grouped data

    Parameters:
    column_name: name of the feature.
    data: dictionary containing feature value per group.
    """
    colors_list = ["red", "blue", "green", "yellow", "pink",
          "purple", "cyan", "black", "olive"]
    colors=colors_list[:len(data)]
    _, ax = plt.subplots()
    for i, (k, v) in enumerate(data.items()):
        ax.bar(v[0], v[1], width=0.2, edgecolor=colors[i],
               align='center', fill=False, label=k)
    ax.set_title(f"Distribution of {column_name} by machine failure")
    ax.set_xticks([0, 1])
    ax.set_ylabel("Count")
    plt.legend()
    plt.show()

def plot_classification_report(
        report:Dict
        ):
    """
    Plot visual representation of classification report.

    Parameters:
    report: classification report as a dictionary.
    """
    labels = list(report.keys())[:-3] 
    metrics = ['precision', 'recall', 'f1-score', 'support']
    data = np.array([[report[label][metric] for metric in metrics] for label in labels])
    _, ax = plt.subplots(figsize=(7, 4))
    cax = ax.matshow(data, cmap='coolwarm')
    ax.set_xticks(range(len(metrics)), metrics)
    ax.set_yticks(range(len(labels)), labels)
    plt.colorbar(cax)
    for (i, j), val in np.ndenumerate(data):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Classes')
    ax.set_title('Classification Report with Support')
    plt.show()