import matplotlib.pyplot as plt

# Sample data for MAE and RMSE values for different features
models = ['ResNet-101', 'VLDSP', 'Combined']
two_layers = [77.19, 61.08, 74.84]
three_layers = [70.12, 42.72, 68]

# Plotting
plt.figure(figsize=(10, 6))

# Bar positions
bar_positions = range(len(models))

# Plotting two_layers values
plt.bar([p - 0.2 for p in bar_positions], two_layers, width=0.3, color='#BA99DC', alpha=0.6, label='2 layers')
# Plotting three_layers values
plt.bar([p + 0.2 for p in bar_positions], three_layers, width=0.3, color='skyblue', alpha=0.6, label='3 layers')

# Annotating bars with scores
for i in range(len(models)):
    plt.text(bar_positions[i] - 0.2, two_layers[i] + 0.1, f'{two_layers[i]:.2f}%', color='black', ha='center')
    plt.text(bar_positions[i] + 0.2, three_layers[i] + 0.1, f'{three_layers[i]:.2f}%', color='black', ha='center')

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('BiLSTM COMPARISON WITH ALL MODELS')
plt.xticks(bar_positions, models)
plt.legend()
plt.grid(False)
# plt.tight_layout()

# Save the graph
plt.savefig('bilstm_all_layers.png')

# Show the graph
plt.show()