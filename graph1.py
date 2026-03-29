import matplotlib.pyplot as plt

# Data
references = ['Sridhar et al. [1]', 'Dahake et al. [2]', 'Das et al. [3]', 
              'Katti et al. [4]', 'Prathap et al. [5]', 'Guo et al. [6]', 
              'Our Method (ResNet-101)', 'Our Method (ResNet-101+VLDSP)']
accuracies = [74.2, 78, 75, 86.7, 71, 74.20, 77.19, 74.84]

# Plot
plt.figure(figsize=(10, 8))  # Adjust figure size
plt.barh(references, accuracies, color='skyblue')
plt.xlabel('Accuracy (%)')
plt.title('Accuracy of Different Models')
plt.xlim(0, 100)

# Add data labels with adjusted spacing
for index, value in enumerate(accuracies):
    plt.text(value, index, f'{value:.2f}%', va='center', ha='left', fontsize=9)  # Adjust fontsize and ha

# Save plot
plt.tight_layout()
plt.savefig('other_models_2.png')
plt.show()