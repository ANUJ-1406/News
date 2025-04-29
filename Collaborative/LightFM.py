#lightFM bpr
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.sparse import coo_matrix
from lightfm import LightFM
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from codecarbon import EmissionsTracker
from datetime import datetime
import os
### Start Carbon Tracker
tracker = EmissionsTracker(measure_power_secs=1, save_to_file=True, output_file="LightFM.csv")
tracker.start()
# --- Load Data ---
train_behaviors = pd.read_csv("/Users/anuj/Downloads/MINDsmall_train/behaviors.tsv", sep='\t', header=None,
                              names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])
valid_behaviors = pd.read_csv("/Users/anuj/Downloads/MINDsmall_dev/behaviors.tsv", sep='\t', header=None,
                              names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])

# --- Build User and News Mappings ---
all_user_ids = pd.concat([train_behaviors['UserID'], valid_behaviors['UserID']]).unique()
user2idx = {uid: i for i, uid in enumerate(all_user_ids)}

news_ids = set()
for df in [train_behaviors, valid_behaviors]:
    for row in df.itertuples():
        for imp in row.Impressions.split():
            nid, _ = imp.split('-')
            news_ids.add(nid)

news2idx = {nid: i for i, nid in enumerate(sorted(news_ids))}

num_users = len(user2idx)
num_items = len(news2idx)

# --- Create Interaction Matrix ---
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

# --- Train LightFM with BPR ---
model_fm = LightFM(no_components=128, loss='bpr', learning_rate=0.05, item_alpha=1e-6, user_alpha=1e-6)
model_fm.fit(interaction_matrix, epochs=30, num_threads=4)

# --- Evaluate ---
all_labels, all_scores = [], []

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
            score = model_fm.predict([uid], [nid_idx], num_threads=1)[0]
        else:
            score = 0.0
        scores.append(score)

    if sum(labels) > 0:
        all_labels.append(labels)
        all_scores.append(scores)

# --- Metrics ---
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
    try:
        auc.append(roc_auc_score(labels, scores))
    except ValueError:
        continue
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
    df = pd.read_csv("LightFM.csv")
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
    with open("emissions/LightFM.txt", "w") as f:
        f.write(report)

except Exception as e:
    print(f"⚠️ Error generating emissions report: {e}")


os.makedirs("results", exist_ok=True)
# Save overall metrics after they are calculated
with open("results/LightFM.txt", "w") as f:
    f.write("Metric\tValue\n")
    f.write(f"AUC\t{np.mean(auc):.6f}\n")
    f.write(f"MRR\t{np.mean(mrr):.6f}\n")
    f.write(f"nDCG@5\t{np.mean(ndcg5):.6f}\n")
    f.write(f"nDCG@10\t{np.mean(ndcg10):.6f}\n")

print("Scores and metrics saved to results/ directory")
import pickle
# Save BPR model
with open('model_fm.pkl', 'wb') as f:
    pickle.dump(model_fm, f)
