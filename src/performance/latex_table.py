import matplotlib.pyplot as plt

# Example data for PathMNIST performance retention
labels = ['ResNet18', 'MobileNetV2', 'EfficientNet-B0']
performance_retained = [97.63, 97.25, 98.54]
size_reduction = [54.40, 86.09, 79.14]

# Define custom colors for both charts
performance_colors = ['#66c2a5', '#fc8d62', '#8da0cb']
size_colors = ['#e78ac3', '#a6d854', '#ffd92f']

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Pie chart for Performance Retained
axs[0].pie(performance_retained, labels=labels, autopct='%1.1f%%',
           startangle=140, colors=performance_colors)
axs[0].set_title('Performance Retained (%) - PathMNIST')

# Pie chart for Size Reduction
axs[1].pie(size_reduction, labels=labels, autopct='%1.1f%%',
           startangle=140, colors=size_colors)
axs[1].set_title('Model Size Reduction (%) - PathMNIST')

plt.tight_layout()
plt.savefig('outputs/performance_piecharts.png', dpi=300)
plt.show()
