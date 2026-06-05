import matplotlib.pyplot as plt
import seaborn as sns
import json

# Data
methods = ['Paper: Blackboard\n+ Claude 4 Opus', 'Ours: DGM-Blackboard\n+ Qwen3.5-35B']
pass_rates = [49.8, 98.3]
colors = ['#E57373', '#4CAF50']

# Setup plot style
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")

# Create bar plot
bars = plt.bar(methods, pass_rates, color=colors, width=0.5)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height}%',
             ha='center', va='bottom', fontweight='bold', fontsize=12)

# Customize plot
plt.title('DS-Bench Unified Data Lake Evaluation\n(Macro Average Pass Rate)', fontsize=14, pad=20, fontweight='bold')
plt.ylabel('Pass Rate (%)', fontsize=12)
plt.ylim(0, 110)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot
output_path = 'data/dsbench_unified_results.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")
