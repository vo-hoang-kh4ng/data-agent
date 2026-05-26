import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'figure.facecolor': '#1e1e24',
    'axes.facecolor': '#1e1e24',
    'text.color': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#b0b0b0',
    'ytick.color': '#b0b0b0',
    'grid.color': '#2d2d38'
})

report_path = r"d:\kh4ng\Data_agent\dgm_agent\outer_eval_report.json"
output_dir = r"d:\kh4ng\Data_agent\dgm_agent\analysis\output"
os.makedirs(output_dir, exist_ok=True)

with open(report_path, "r", encoding="utf-8") as f:
    data = json.load(f)

results = data.get("task_results", [])

# 1. Calculate Pass Rates by Language
lang_data = {}
for r in results:
    lang = r.get("language", "unknown").upper()
    status = r.get("status", "FAIL")
    is_pass = r.get("pass", False)
    
    if lang not in lang_data:
        lang_data[lang] = {"total": 0, "pass": 0, "skip": 0}
        
    if status == "SKIP":
        lang_data[lang]["skip"] += 1
    else:
        lang_data[lang]["total"] += 1
        if is_pass:
            lang_data[lang]["pass"] += 1

languages = sorted(list(lang_data.keys()))
pass_rates = []
totals = []
passes = []

for lang in languages:
    tot = lang_data[lang]["total"]
    pas = lang_data[lang]["pass"]
    rate = (pas / tot * 100) if tot > 0 else 0.0
    pass_rates.append(rate)
    totals.append(tot)
    passes.append(pas)

# Plot 1: Pass Rate by Language
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#8E44AD', '#16A085']
if len(languages) > len(colors):
    colors = colors * 2
colors = colors[:len(languages)]

bars = ax.bar(languages, pass_rates, color=colors, edgecolor='#2d2d38', width=0.6)

# Add values on top of bars
for bar, pas, tot in zip(bars, passes, totals):
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%\n({pas}/{tot})',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='#ffffff')

ax.set_ylim(0, 115)
ax.set_ylabel('Pass Rate (%)', fontweight='bold', fontsize=14)
ax.set_xlabel('Programming Language', fontweight='bold', fontsize=14)
ax.set_title('Polyglot Benchmark Pass Rate by Language', fontsize=16, fontweight='bold', color='#ffffff', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "polyglot_pass_rates.png"), dpi=300, facecolor='#1e1e24')
plt.close()
print("Saved polyglot_pass_rates.png")

# Plot 2: Detailed Task Status Matrix Grid
# Sort tasks by language, then task_id
sorted_tasks = sorted(results, key=lambda x: (x.get("language", ""), x.get("task_id", "")))

# Chunk tasks into grids for visualization
num_tasks = len(sorted_tasks)
cols = 4
rows = int(np.ceil(num_tasks / cols))

fig, ax = plt.subplots(figsize=(16, rows * 0.75))
ax.set_facecolor('#1e1e24')

for idx, task in enumerate(sorted_tasks):
    r = idx // cols
    c = idx % cols
    
    status = task.get("status", "FAIL")
    task_id = task.get("task_id", "")
    lang = task.get("language", "").upper()
    
    if status == "PASS":
        color = '#2e7d32' # Green
        text_color = '#ffffff'
    elif status == "SKIP":
        color = '#757575' # Gray
        text_color = '#ffffff'
    else:
        color = '#c62828' # Red
        text_color = '#ffffff'
        
    # Draw a colored bounding box for each task
    rect = plt.Rectangle((c - 0.45, rows - r - 0.8), 0.9, 0.7, facecolor=color, edgecolor='#2d2d38', zorder=2)
    ax.add_patch(rect)
    
    # Label inside box
    short_task_name = task_id.split("__")[-1] if "__" in task_id else task_id
    ax.text(c, rows - r - 0.45, f"[{lang}]", ha='center', va='center', fontsize=9, color='#e0e0e0', fontweight='bold', zorder=3)
    ax.text(c, rows - r - 0.6, short_task_name, ha='center', va='center', fontsize=10, color=text_color, fontweight='bold', zorder=3)

ax.set_xlim(-0.6, cols - 0.4)
ax.set_ylim(-0.2, rows)
ax.axis('off')
ax.set_title('Transparent Polyglot Task Status Matrix', fontsize=18, fontweight='bold', color='#ffffff', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "task_status_matrix.png"), dpi=300, facecolor='#1e1e24')
plt.close()
print("Saved task_status_matrix.png")
