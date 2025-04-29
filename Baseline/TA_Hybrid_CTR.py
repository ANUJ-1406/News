
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import json
from sklearn.metrics import roc_auc_score
from codecarbon import EmissionsTracker
from datetime import datetime
import os
### Start Carbon Tracker
tracker = EmissionsTracker(measure_power_secs=1, save_to_file=True, output_file="TA_Hybrid_CTR_72.csv")
tracker.start()
### Configurable Hyperparameters
TIME_WINDOW = 3 * 24 * 60 * 60   # 3 days window
ALPHA = 0.9                      # blending weight for time-aware vs global CTR

### Load Data
train_behaviors = pd.read_csv("/Users/anuj/Downloads/MINDsmall_train/behaviors.tsv", sep='\t', header=None,
                              names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])
valid_behaviors = pd.read_csv("/Users/anuj/Downloads/MINDsmall_dev/behaviors.tsv", sep='\t', header=None,
                              names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])


### Convert timestamp to numeric seconds (assuming it's sortable)
train_behaviors['Timestamp'] = pd.to_datetime(train_behaviors['Time']).astype(int) // 10**9
valid_behaviors['Timestamp'] = pd.to_datetime(valid_behaviors['Time']).astype(int) // 10**9

first_valid_time = valid_behaviors.iloc[0]['Timestamp']

### Filter training to rolling window only
train_behaviors = train_behaviors[(train_behaviors['Timestamp'] >= first_valid_time - TIME_WINDOW) &
                                  (train_behaviors['Timestamp'] < first_valid_time)]

### Global Click Counts
click_counts = defaultdict(int)
view_counts = defaultdict(int)
news_stats = defaultdict(lambda: {'clicks': deque(), 'impressions': deque()})

for row in train_behaviors.itertuples():
    for impression in row.Impressions.split():
        news_id, label = impression.split('-')
        view_counts[news_id] += 1
        news_stats[news_id]['impressions'].append(row.Timestamp)
        if label == '1':
            click_counts[news_id] += 1
            news_stats[news_id]['clicks'].append(row.Timestamp)
# Global CTR
ctr_global = {nid: click_counts[nid]/view_counts[nid] for nid in view_counts}


           
print("evaluating")
### Helper Functions
def update_rolling_stats(current_time):
    for nid in list(news_stats.keys()):
        while news_stats[nid]['clicks'] and news_stats[nid]['clicks'][0] < current_time - TIME_WINDOW:
            news_stats[nid]['clicks'].popleft()
        while news_stats[nid]['impressions'] and news_stats[nid]['impressions'][0] < current_time - TIME_WINDOW:
            news_stats[nid]['impressions'].popleft()
        if not news_stats[nid]['clicks'] and not news_stats[nid]['impressions']:
            del news_stats[nid]

### Evaluation Storage
all_labels, all_scores = [], []
time_aware_ctr_dict={}
for row in valid_behaviors.itertuples():
    current_time = row.Timestamp
    update_rolling_stats(current_time)

    labels, scores = [], []

    for impression in row.Impressions.split():
        news_id, label = impression.split('-')
        labels.append(int(label))

        global_ctr = ctr_global.get(news_id, 0.0)
        stats = news_stats.get(news_id, {'clicks': deque(), 'impressions': deque()})
        time_ctr = len(stats['clicks']) / len(stats['impressions']) if len(stats['impressions']) > 0 else 0.0
        time_aware_ctr_dict[news_id] = time_ctr
        final_score = ALPHA * time_ctr + (1 - ALPHA) * global_ctr
        scores.append(final_score)

        news_stats[news_id]['impressions'].append(current_time)
        if label == '1':
            news_stats[news_id]['clicks'].append(current_time)

    all_labels.append(labels)
    all_scores.append(scores)

### Metrics
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
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return np.sum(gains / discounts)

def ndcg_score(labels, scores, k):
    dcg = dcg_score(labels, scores, k)
    ideal_dcg = dcg_score(labels, labels, k)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

mrr, ndcg5, ndcg10, auc = [], [], [], []

for labels, scores in zip(all_labels, all_scores):
    if sum(labels) == 0:
        continue

    auc.append(roc_auc_score(labels, scores))
    mrr.append(mrr_score(labels, scores))
    ndcg5.append(ndcg_score(labels, scores, 5))
    ndcg10.append(ndcg_score(labels, scores, 10))

print(f"AUC: {np.mean(auc):.4f}")
print(f"MRR: {np.mean(mrr):.4f}")
print(f"nDCG@5: {np.mean(ndcg5):.4f}")
print(f"nDCG@10: {np.mean(ndcg10):.4f}")

import pickle

# Save the model
with open('ctr_global.pkl', 'wb') as f:
    pickle.dump(ctr_global, f)
with open('time_aware_ctr.pkl', 'wb') as f:
    pickle.dump(time_aware_ctr_dict, f)

# Stop tracking
emissions = tracker.stop()
# --- Generate and print report ---
try:
    df = pd.read_csv("TA_Hybrid_CTR_72.csv")
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
    with open("emissions/TA_Hybrid_CTR_72.txt", "w") as f:
        f.write(report)

except Exception as e:
    print(f"⚠️ Error generating emissions report: {e}")


print("Saving scores to file...")

# Create directory for results if it doesn't exist
os.makedirs("results", exist_ok=True)
# Save overall metrics after they are calculated
with open("results/TA_Hybrid_CTR_72.txt", "w") as f:
    f.write("Metric\tValue\n")
    f.write(f"AUC\t{np.mean(auc):.6f}\n")
    f.write(f"MRR\t{np.mean(mrr):.6f}\n")
    f.write(f"nDCG@5\t{np.mean(ndcg5):.6f}\n")
    f.write(f"nDCG@10\t{np.mean(ndcg10):.6f}\n")

print("Scores and metrics saved to results/ directory")

