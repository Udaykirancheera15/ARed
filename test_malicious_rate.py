# Generate comparison chart
import matplotlib.pyplot as plt
import numpy as np

models = ['Llama-3-8B', 'Mistral-7B', 'Llama-2-7B', 'Gemma-2B']
our_rates = [98.57, 97.14, 67.14, 41.43]
paper_rates = [61, 73, 65, 83]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, our_rates, width, label='Ours', color='steelblue', edgecolor='black')
bars2 = ax.bar(x + width/2, paper_rates, width, label='Paper', color='lightcoral', edgecolor='black')

ax.set_ylabel('Success Rate (%)')
ax.set_xlabel('Target LLM')
ax.set_title('AUTORED Performance: Ours vs Paper')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0, 105)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

for bar, rate in zip(bars1, our_rates):
    ax.annotate(f'{rate}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
               xytext=(0, 3), textcoords='offset points', ha='center', fontweight='bold')

for bar, rate in zip(bars2, paper_rates):
    ax.annotate(f'{rate}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
               xytext=(0, 3), textcoords='offset points', ha='center')

plt.tight_layout()
plt.savefig('outputs/figures/slide23_comparison.png', dpi=300)
print("✅ Slide 23 figure saved")
