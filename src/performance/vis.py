# src/performance/vis.py

import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_performance_vs_size():
    # Dataset-wise model data
    data = {
        "Dataset": [
            "PathMNIST", "PathMNIST", "PathMNIST",
            "OCTMNIST", "OCTMNIST", "OCTMNIST",
            "OrganAMNIST", "OrganAMNIST", "OrganAMNIST",
            "ChestMNIST", "ChestMNIST", "ChestMNIST",
            "DermaMNIST", "DermaMNIST", "DermaMNIST",
        ],
        "Model": [
            "ResNet18", "MobileNetV2", "EfficientNet-B0",
            "ResNet18", "MobileNetV2", "EfficientNet-B0",
            "ResNet18", "MobileNetV2", "EfficientNet-B0",
            "ResNet18", "MobileNetV2", "EfficientNet-B0",
            "ResNet18", "MobileNetV2", "EfficientNet-B0",
        ],
        "Performance Retention (%)": [
            97.63, 97.25, 98.54,
            99.71, 98.42, 95.83,
            99.10, 98.26, 98.42,
            92.94, 89.80, 89.99,
            94.81, 96.05, 94.81,
        ],
        "Model Size (MB)": [
            44.6, 13.6, 20.4,
            44.6, 13.6, 20.4,
            44.6, 13.6, 20.4,
            44.6, 13.6, 20.4,
            44.6, 13.6, 20.4,
        ]
    }

    import pandas as pd
    df = pd.DataFrame(data)

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 7))

    # Scatter plot: Model Size vs Performance Retention
    markers = {"ResNet18": "o", "MobileNetV2": "s", "EfficientNet-B0": "D"}
    palette = {"PathMNIST": "#1f77b4", "OCTMNIST": "#ff7f0e", "OrganAMNIST": "#2ca02c",
               "ChestMNIST": "#d62728", "DermaMNIST": "#9467bd"}

    for model in df['Model'].unique():
        subset = df[df['Model'] == model]
        plt.scatter(
            subset['Model Size (MB)'],
            subset['Performance Retention (%)'],
            s=100,
            alpha=0.8,
            marker=markers[model],
            label=model
        )
        # Add text labels for datasets
        for _, row in subset.iterrows():
            plt.text(row['Model Size (MB)']+0.5, row['Performance Retention (%)']-0.3, row['Dataset'], fontsize=9)

    plt.title('Knowledge Distillation: Performance Retention vs Model Size on MedMNIST', fontsize=16)
    plt.xlabel('Model Size (MB)', fontsize=14)
    plt.ylabel('Performance Retention (%)', fontsize=14)
    plt.ylim(85, 102)
    plt.xlim(10, 50)
    plt.legend(title='Student Models')
    plt.tight_layout()

    # Create outputs directory if not exists
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/performance_vs_size.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_performance_vs_size()
