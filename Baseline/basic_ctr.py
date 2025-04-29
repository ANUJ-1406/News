### Super Baseline Recommender (MIND Small)

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from datetime import datetime
from codecarbon import EmissionsTracker

import os
### Start Carbon Tracker
tracker = EmissionsTracker(measure_power_secs=1, save_to_file=True, output_file="basic_CTR.csv")
tracker.start()
### Load News Metadata
news = pd.read_csv(r"/Users/anuj/Downloads/MINDsmall_train/news.tsv", sep='\t', header=None,
                   names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'])
news_category = dict(zip(news.NewsID, news.Category))

### Load Train Behaviors and Compute CTR Stats
train_behaviors = pd.read_csv(r"/Users/anuj/Downloads/MINDsmall_train/behaviors.tsv", sep='\t', header=None,
                              names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])

# CTR counts
click_counts, impression_counts = defaultdict(int), defaultdict(int)
category_click_counts, category_impression_counts = defaultdict(int), defaultdict(int)
user_category_click_counts = defaultdict(lambda: defaultdict(int))
user_category_imp_counts = defaultdict(lambda: defaultdict(int))

for row in train_behaviors.itertuples():
    user_id = row.UserID
    time = datetime.strptime(row.Time, '%m/%d/%Y %I:%M:%S %p')
    impressions = row.Impressions.split()
    for imp in impressions:
        news_id, label = imp.split('-')
        category = news_category.get(news_id, 'Unknown')
        label = int(label)
        
        impression_counts[news_id] += 1
        category_impression_counts[category] += 1
        user_category_imp_counts[user_id][category] += 1
        
        if label == 1:
            click_counts[news_id] += 1
            category_click_counts[category] += 1
            user_category_click_counts[user_id][category] += 1

# CTR Calculations
def compute_ctr(clicks, impressions):
    return clicks / impressions if impressions > 0 else 0.0

ctr = {nid: compute_ctr(click_counts[nid], impression_counts[nid]) for nid in impression_counts}
category_ctr = {cat: compute_ctr(category_click_counts[cat], category_impression_counts[cat]) for cat in category_impression_counts}

### Load Validation Behaviors
valid_behaviors = pd.read_csv(r"/Users/anuj/Downloads/MINDsmall_dev/behaviors.tsv", sep='\t', header=None,
                              names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])

### Evaluation Metrics
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

### Super Baseline Scoring
w1, w2, w3 = 0.5, 0.3, 0.2  # Tunable weights
time_decay_lambda = 0.8  # Decay rate

mrr, ndcg5, ndcg10, auc = [], [], [], []

for row in valid_behaviors.itertuples():
    user_id = row.UserID
    time = datetime.strptime(row.Time, '%m/%d/%Y %I:%M:%S %p')
    history = set(row.History.split() if pd.notna(row.History) else [])
    impressions = row.Impressions.split()
    labels, scores = [], []

    for imp in impressions:
        news_id, label = imp.split('-')
        category = news_category.get(news_id, 'Unknown')
        labels.append(int(label))

        if news_id in history:
            score = -1e9  # Exclude already read
        else:
            user_cat_ctr = compute_ctr(user_category_click_counts[user_id][category], 
                                       user_category_imp_counts[user_id][category])
            cat_ctr = category_ctr.get(category, 0)
            global_ctr = ctr.get(news_id, 0)

            # Time decay based on hours since midnight
            hours_since_midnight = time.hour + time.minute / 60.0
            decay = np.exp(-time_decay_lambda * hours_since_midnight / 24.0)

            score = (w1 * user_cat_ctr + w2 * cat_ctr + w3 * global_ctr) * decay

        scores.append(score)

    if sum(labels) == 0:
        continue

    try:
        auc.append(roc_auc_score(labels, scores))
    except:
        pass

    mrr.append(mrr_score(labels, scores))
    ndcg5.append(ndcg_score(labels, scores, 5))
    ndcg10.append(ndcg_score(labels, scores, 10))


# Stop tracking
emissions = tracker.stop()
# --- Generate and print report ---
try:
    df = pd.read_csv("basic_CTR.csv")
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
    with open("emissions/basic_CTR.txt", "w") as f:
        f.write(report)

except Exception as e:
    print(f"⚠️ Error generating emissions report: {e}")


print("Saving scores to file...")

# Create directory for results if it doesn't exist
os.makedirs("results", exist_ok=True)
# Save overall metrics after they are calculated
with open("results/basic_CTR.txt", "w") as f:
    f.write("Metric\tValue\n")
    f.write(f"AUC\t{np.mean(auc):.6f}\n")
    f.write(f"MRR\t{np.mean(mrr):.6f}\n")
    f.write(f"nDCG@5\t{np.mean(ndcg5):.6f}\n")
    f.write(f"nDCG@10\t{np.mean(ndcg10):.6f}\n")

print("Scores and metrics saved to results/ directory")