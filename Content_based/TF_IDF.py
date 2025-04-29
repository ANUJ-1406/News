import numpy as np
import pandas as pd
import pickle
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
from codecarbon import EmissionsTracker
from datetime import datetime
import os
### Start Carbon Tracker
tracker = EmissionsTracker(measure_power_secs=1, save_to_file=True, output_file="tfidf.csv")
tracker.start()
# Start timer
start_time = time.time()

# Load data
news_df = pd.read_csv("/Users/anuj/Downloads/MINDsmall_train/news.tsv", sep='\t', header=None,
                      names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'])
behavior_df = pd.read_csv("/Users/anuj/Downloads/MINDsmall_train/behaviors.tsv", sep='\t', header=None,
                          names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])
news_dev = pd.read_csv("/Users/anuj/Downloads/MINDsmall_dev/news.tsv", sep='\t', header=None,
                       names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'])
behaviors_dev = pd.read_csv("/Users/anuj/Downloads/MINDsmall_dev/behaviors.tsv", sep='\t', header=None,
                            names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])

# Merge news
news_df_full = pd.concat([news_df, news_dev]).drop_duplicates(subset="NewsID")
news_df_full['NewsID'] = news_df_full['NewsID'].astype(str)
for col in ['Title', 'Abstract', 'Category', 'SubCategory']:
    news_df_full[col] = news_df_full[col].fillna('')
news_df_full['content'] = (news_df_full['Title'] + ' ' + news_df_full['Abstract'] + ' ' + 
                           news_df_full['Category'] + ' ' + news_df_full['SubCategory'])

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)
tfidf_matrix = tfidf_vectorizer.fit_transform(news_df_full['content'])
news_id_to_index = {news_id: idx for idx, news_id in enumerate(news_df_full['NewsID'])}

# Create user profiles
user_profiles = {}
for idx, row in behavior_df.iterrows():
    user_id = str(row['UserID'])
    history = row['History']
    if pd.isna(history) or not history:
        continue
    history_articles = history.strip().split()
    indices = [news_id_to_index[aid] for aid in history_articles if aid in news_id_to_index]
    if indices:
        user_vector = np.array(tfidf_matrix[indices].mean(axis=0)).flatten()
        user_profiles[user_id] = user_vector

# Evaluate on dev set
all_aucs, all_mrrs, all_ndcg5, all_ndcg10 = [], [], [], []
skipped_users = 0
total_impressions = 0

for idx, row in behaviors_dev.iterrows():
    user_id = str(row['UserID'])
    if user_id not in user_profiles:
        skipped_users += 1
        continue
    
    impressions = row['Impressions']
    if pd.isna(impressions) or not impressions:
        continue
    
    items = []
    labels = []
    for item in impressions.strip().split():
        if '-' in item:
            news_id, clicked = item.split('-')
            if news_id in news_id_to_index:
                items.append(news_id_to_index[news_id])
                labels.append(int(clicked))
    
    if not items or len(set(labels)) < 2:
        continue
    
    user_vec = user_profiles[user_id].reshape(1, -1)
    item_vecs = tfidf_matrix[items]
    scores = cosine_similarity(user_vec, item_vecs)[0]
    
    # Compute metrics
    auc = roc_auc_score(labels, scores)
    
    sorted_pairs = sorted(zip(scores, labels), reverse=True)
    mrr = 0
    for rank, (_, label) in enumerate(sorted_pairs):
        if label == 1:
            mrr = 1.0 / (rank + 1)
            break
    
    def dcg_at_k(rels, k):
        rels = np.asfarray(rels)[:k]
        return np.sum(rels / np.log2(np.arange(2, rels.size + 2))) if rels.size else 0.

    def ndcg_at_k(labels_sorted, k):
        ideal = sorted(labels_sorted, reverse=True)
        return dcg_at_k(labels_sorted, k) / dcg_at_k(ideal, k) if dcg_at_k(ideal, k) != 0 else 0.

    labels_sorted = [label for _, label in sorted_pairs]
    ndcg5 = ndcg_at_k(labels_sorted, 5)
    ndcg10 = ndcg_at_k(labels_sorted, 10)

    all_aucs.append(auc)
    all_mrrs.append(mrr)
    all_ndcg5.append(ndcg5)
    all_ndcg10.append(ndcg10)
    total_impressions += len(labels)

# Final metrics
print(f"\nEvaluation Results:")
print(f"Evaluated on {len(all_aucs)} impression sessions")
print(f"Total impressions: {total_impressions}")
print(f"Skipped users (no profile): {skipped_users}")
print(f"AUC: {np.mean(all_aucs):.4f}")
print(f"MRR: {np.mean(all_mrrs):.4f}")
print(f"NDCG@5: {np.mean(all_ndcg5):.4f}")
print(f"NDCG@10: {np.mean(all_ndcg10):.4f}")

# Save artifacts
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
with open('user_profiles.pkl', 'wb') as f:
    pickle.dump(user_profiles, f)
with open('tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)
with open('news_id_to_index.pkl', 'wb') as f:
    pickle.dump(news_id_to_index, f)

emissions = tracker.stop()
# --- Generate and print report ---
try:
    df = pd.read_csv("tfidf.csv")
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
    with open("emissions/tfidf.txt", "w") as f:
        f.write(report)

except Exception as e:
    print(f"⚠️ Error generating emissions report: {e}")

# Save overall metrics after they are calculated
with open("results/tfidf.txt", "w") as f:
    f.write("Metric\tValue\n")
    f.write(f"AUC\t{np.mean(all_aucs):.6f}\n")
    f.write(f"MRR\t{np.mean(all_mrrs):.6f}\n")
    f.write(f"nDCG@5\t{np.mean(all_ndcg5):.6f}\n")
    f.write(f"nDCG@10\t{np.mean(all_ndcg10):.6f}\n")

print("Scores and metrics saved to results/ directory")



