import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import numpy as np
from codecarbon import EmissionsTracker
from datetime import datetime
import os
### Start Carbon Tracker
tracker = EmissionsTracker(measure_power_secs=1, save_to_file=True, output_file="R_P_12.csv")
tracker.start()

# Configurable parameters
time_window_hours = 12  # can adjust this value
decay_lambda = 1     # decay rate (higher = faster decay)

### Step 1: Load Train Behaviors and Count Clicks

train_behaviors = pd.read_csv(r"/Users/anuj/Downloads/MINDsmall_train/behaviors.tsv", sep='\t', header=None,
                              names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])
train_behaviors['Time'] = pd.to_datetime(train_behaviors['Time'])

# Calculate click counts with optional time decay
click_counts = defaultdict(float)
current_time = train_behaviors['Time'].max()

for row in train_behaviors.itertuples():
    impressions = row.Impressions.split()
    user_history = set(row.History.split() if isinstance(row.History, str) else [])
    time_diff = (current_time - row.Time).total_seconds() / 3600.0  # hours
    
    if time_diff > time_window_hours:
        continue

    decay_weight = np.exp(-decay_lambda * time_diff)
    
    for impression in impressions:
        news_id, label = impression.split('-')
        if news_id in user_history:
            continue
        if label == '1':
            click_counts[news_id] += decay_weight

print(f"Top 5 weighted clicked news articles:")
print(sorted(click_counts.items(), key=lambda x: x[1], reverse=True)[:5])


### Step 2: Load Validation Behaviors and Score by Popularity

valid_behaviors = pd.read_csv(r"/Users/anuj/Downloads/MINDsmall_dev/behaviors.tsv", sep='\t', header=None,
                              names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])

# Prepare labels and scores
all_labels, all_scores = [], []

for row in valid_behaviors.itertuples():
    impressions = row.Impressions.split()
    user_history = set(row.History.split() if isinstance(row.History, str) else [])

    labels, scores = [], []

    for impression in impressions:
        news_id, label = impression.split('-')
        labels.append(int(label))
        score = click_counts.get(news_id, 0.0)
        scores.append(score)

    all_labels.append(labels)
    all_scores.append(scores)


### Step 3: Evaluation Metrics

def mrr_score(labels, scores):
    order = np.argsort(scores)[::-1]
    labels = np.array(labels)[order]
    for idx, label in enumerate(labels):
        if label == 1:
            return 1.0 / (idx + 1)
    return 0.0

def dcg_score(labels, scores, k):
    order = np.argsort(scores)[::-1][:k]
    gains = np.array(labels)[order]
    discounts = np.log2(np.arange(2, len(gains) + 2))  # dynamic length
    return np.sum(gains / discounts)


def ndcg_score(labels, scores, k):
    dcg = dcg_score(labels, scores, k)
    ideal_dcg = dcg_score(labels, labels, k)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

mrr, ndcg5, ndcg10, auc = [], [], [], []


for labels, scores in zip(all_labels, all_scores):
    if sum(labels) == 0:
        continue

    try:
        auc.append(roc_auc_score(labels, scores))
    except:
        pass

    mrr.append(mrr_score(labels, scores))
    ndcg5.append(ndcg_score(labels, scores, 5))
    ndcg10.append(ndcg_score(labels, scores, 10))


print(f"AUC: {np.mean(auc):.4f}")
print(f"MRR: {np.mean(mrr):.4f}")
print(f"nDCG@5: {np.mean(ndcg5):.4f}")
print(f"nDCG@10: {np.mean(ndcg10):.4f}")


# Stop tracking
emissions = tracker.stop()
# --- Generate and print report ---
try:
    df = pd.read_csv("R_P_12.csv")
    emissions_data = df.iloc[-1]

    duration_hr = emissions_data['duration'] / 3600
    energy_kwh = emissions_data['energy_consumed']
    cpu_power = emissions_data['cpu_power']

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""\
📄 Emissions Report – {timestamp}
====================================
🌱 Total Emissions:     {emissions:.6f} kg CO2eq

🕒 Duration:            {duration_hr:.2f} hours
⚡ Energy Consumed:     {energy_kwh:.4f} kWh
🧠 CPU Power:           {cpu_power:.2f} W

🌍 Machine:             MacBook Air (CPU Only)
====================================
"""

    print(report)

    os.makedirs("emissions", exist_ok=True)
    with open("emissions/R_P_12.txt", "w") as f:
        f.write(report)

except Exception as e:
    print(f"⚠️ Error generating emissions report: {e}")


print("Saving scores to file...")

# Create directory for results if it doesn't exist
os.makedirs("results", exist_ok=True)

# Save overall metrics after they are calculated
with open("results/R_P_12.txt", "w") as f:
    f.write("Metric\tValue\n")
    f.write(f"AUC\t{np.mean(auc):.6f}\n")
    f.write(f"MRR\t{np.mean(mrr):.6f}\n")
    f.write(f"nDCG@5\t{np.mean(ndcg5):.6f}\n")
    f.write(f"nDCG@10\t{np.mean(ndcg10):.6f}\n")

print("Scores and metrics saved to results/ directory")