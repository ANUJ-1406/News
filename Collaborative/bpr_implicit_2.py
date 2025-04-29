#bpr implicit
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.sparse import coo_matrix
from implicit.bpr import BayesianPersonalizedRanking
from sklearn.metrics import roc_auc_score
from codecarbon import EmissionsTracker
from datetime import datetime
import os
### Start Carbon Tracker
tracker = EmissionsTracker(measure_power_secs=1, save_to_file=True, output_file="bpr_2.csv")
tracker.start()
# Load Data
train_behaviors = pd.read_csv("/Users/anuj/Downloads/MINDsmall_train/behaviors.tsv", sep='\t', header=None,
                              names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])
valid_behaviors = pd.read_csv("/Users/anuj/Downloads/MINDsmall_dev/behaviors.tsv", sep='\t', header=None,
                              names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])

# Build User and News ID mappings
unique_user_ids = train_behaviors['UserID'].unique()
user2idx = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
news_ids = set()

for row in train_behaviors.itertuples():
    for impression in row.Impressions.split():
        news_id, _ = impression.split('-')
        news_ids.add(news_id)

unique_news_ids = sorted(list(news_ids))
news2idx = {news_id: idx for idx, news_id in enumerate(unique_news_ids)}

num_users = len(user2idx)
num_items = len(news2idx)

# Build sparse interaction matrix (clicks only)
rows, cols, data = [], [], []

for row in train_behaviors.itertuples():
    uid = user2idx[row.UserID]
    for imp in row.Impressions.split():
        nid, label = imp.split('-')
        if label == '1':
            rows.append(uid)
            cols.append(news2idx[nid])
            data.append(1)

interaction_matrix = coo_matrix((data, (rows, cols)), shape=(num_users, num_items))

# Train implicit BPR model
model = BayesianPersonalizedRanking(factors=128, learning_rate=0.02, regularization=0.01, iterations=20)
model.fit(interaction_matrix)

# Evaluate on validation set
all_labels, all_scores = [], []

user_factors = model.user_factors
item_factors = model.item_factors

for row in valid_behaviors.itertuples():
    if row.UserID not in user2idx:
        continue
    uid = user2idx[row.UserID]
    labels, scores = [], []

    for imp in row.Impressions.split():
        nid, label = imp.split('-')
        labels.append(int(label))
        if nid in news2idx:
            nid_idx = news2idx[nid]
            score = np.dot(user_factors[uid], item_factors[nid_idx])
        else:
            score = 0.0
        scores.append(score)

    if sum(labels) > 0:
        all_labels.append(labels)
        all_scores.append(scores)

# Metrics
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
    auc.append(roc_auc_score(labels, scores))
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
    df = pd.read_csv("bpr_2.csv")
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
    with open("emissions/bpr_2.txt", "w") as f:
        f.write(report)

except Exception as e:
    print(f"⚠️ Error generating emissions report: {e}")


os.makedirs("results", exist_ok=True)
# Save overall metrics after they are calculated
with open("results/bpr_2.txt", "w") as f:
    f.write("Metric\tValue\n")
    f.write(f"AUC\t{np.mean(auc):.6f}\n")
    f.write(f"MRR\t{np.mean(mrr):.6f}\n")
    f.write(f"nDCG@5\t{np.mean(ndcg5):.6f}\n")
    f.write(f"nDCG@10\t{np.mean(ndcg10):.6f}\n")

print("Scores and metrics saved to results/ directory")
import pickle
# Save BPR model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Also save user2idx and news2idx mappings
with open('user2idx.pkl', 'wb') as f:
    pickle.dump(user2idx, f)

with open('news2idx.pkl', 'wb') as f:
    pickle.dump(news2idx, f)