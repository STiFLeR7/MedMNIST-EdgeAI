import matplotlib.pyplot as plt

# Data
datasets = ['PathMNIST', 'ChestMNIST', 'OrganAMNIST', 'OCTMNIST', 'DermaMNIST']
models = ['ResNet18', 'MobileNetV2', 'EfficientNet-B0']
sizes = {
    'ResNet18': 44.6,
    'MobileNetV2': 13.6,
    'EfficientNet-B0': 20.4
}
data = {
    'ResNet18': [97.63, 92.94, 99.10, 99.71, 94.81],
    'MobileNetV2': [97.25, 89.80, 98.26, 98.42, 96.05],
    'EfficientNet-B0': [98.54, 89.99, 98.42, 95.83, 94.81]
}

# Colors and markers
colors = {'ResNet18': '#1f77b4', 'MobileNetV2': '#ff7f0e', 'EfficientNet-B0': '#2ca02c'}
markers = {'ResNet18': 'o', 'MobileNetV2': 's', 'EfficientNet-B0': 'D'}

plt.figure(figsize=(12, 7))

# Plot each model
for model in models:
    x = [sizes[model]] * len(datasets)
    y = data[model]
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.scatter(xi, yi, label=f'{model} - {datasets[i]}' if i == 0 else "", 
                    color=colors[model], marker=markers[model], s=100, edgecolor='black')
        plt.text(xi + 0.3, yi, datasets[i], fontsize=9, va='center', ha='left')

# Axis config
plt.xlabel('Model Size (MB)', fontsize=12)
plt.ylabel('Performance Retention (%)', fontsize=12)
plt.title('Knowledge Distillation: Performance Retention vs Model Size on MedMNIST', fontsize=14)
plt.xlim(10, 50)
plt.ylim(85, 102)
plt.grid(True, linestyle='--', alpha=0.5)

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='ResNet18', markerfacecolor=colors['ResNet18'], markersize=10, markeredgecolor='black'),
    Line2D([0], [0], marker='s', color='w', label='MobileNetV2', markerfacecolor=colors['MobileNetV2'], markersize=10, markeredgecolor='black'),
    Line2D([0], [0], marker='D', color='w', label='EfficientNet-B0', markerfacecolor=colors['EfficientNet-B0'], markersize=10, markeredgecolor='black'),
]
plt.legend(handles=legend_elements, title='Student Models', loc='lower left', fontsize=10)

# Save and show
plt.tight_layout()
plt.savefig('outputs/performance_vs_size_cleaned.png', dpi=300)
plt.show()
