### not a good method as it takes a lot of time to train and also a lot of carbon emission with auc increase of 0.003


import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os

# ----------------------------
# 📂 Load News and Behavior Data
# ----------------------------
news = pd.read_csv("/Users/anuj/Downloads/MINDsmall_train/news.tsv", sep='\t', header=None)
news.columns = ["itemId", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
news_test = pd.read_csv("/Users/anuj/Downloads/MINDsmall_dev/news.tsv", sep='\t', header=None)
news_test.columns = news.columns
news = pd.concat([news, news_test], ignore_index=True)

news = news.drop_duplicates('itemId')
itemId_map = {k: v for v, k in enumerate(news.itemId.unique(), start=1)}
ind2item = {v: k for k, v in itemId_map.items()}
news['itemIdx'] = news['itemId'].map(itemId_map)

# ----------------------------
# 🧑‍🤝‍🧑 Process Behaviors
# ----------------------------
def parse_behaviors(path, user2ind):
    data = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            userId = parts[1]
            imprs = parts[-1].split()
            if userId not in user2ind:
                user2ind[userId] = len(user2ind) + 1
            userIdx = user2ind[userId]
            for impr in imprs:
                nid, label = impr.split("-")
                if nid in itemId_map:
                    data.append((userIdx, itemId_map[nid], int(label)))
    return pd.DataFrame(data, columns=['userIdx', 'itemIdx', 'label'])

user2ind = {}
train_behaviors = parse_behaviors("/Users/anuj/Downloads/MINDsmall_train/behaviors.tsv", user2ind)
test_behaviors = parse_behaviors("/Users/anuj/Downloads/MINDsmall_dev/behaviors.tsv", user2ind)

# ----------------------------
# 🔍 News CF Dataset
# ----------------------------
class NewsDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'userIdx': torch.tensor(row.userIdx, dtype=torch.long),
            'itemIdx': torch.tensor(row.itemIdx, dtype=torch.long),
            'label': torch.tensor(row.label, dtype=torch.float)
        }

train_loader = DataLoader(NewsDataset(train_behaviors), batch_size=1024, shuffle=True)
test_loader = DataLoader(NewsDataset(test_behaviors), batch_size=1024, shuffle=False)

# ----------------------------
# 🧠 Collaborative Filtering Model
# ----------------------------
class NewsMF(pl.LightningModule):
    def __init__(self, num_users, num_items, dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users + 1, dim)
        self.item_emb = nn.Embedding(num_items + 1, dim)

    def forward(self, user, item):
        u = self.user_emb(user)
        v = self.item_emb(item)
        return (u * v).sum(1)

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch['userIdx'], batch['itemIdx'])
        loss = F.binary_cross_entropy_with_logits(pred, batch['label'])
        return loss

    def validation_step(self, batch, batch_idx):
        pred = torch.sigmoid(self.forward(batch['userIdx'], batch['itemIdx']))
        self.log('val_auc', roc_auc_score(batch['label'].cpu(), pred.cpu().detach()), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

model = NewsMF(num_users=max(user2ind.values()), num_items=len(itemId_map))

# ----------------------------
# 🚀 Train the model
# ----------------------------
trainer = pl.Trainer(max_epochs=5, accelerator='auto', logger=False)
trainer.fit(model, train_loader)
# ----------------------------
# ✅ Evaluate on Test Set
# ----------------------------
def evaluate_model(model, loader):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            preds = torch.sigmoid(model(batch['userIdx'], batch['itemIdx']))
            all_labels.extend(batch['label'].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    auc = roc_auc_score(all_labels, all_preds)
    print(f"Test AUC: {auc:.4f}")
    return auc

# ----------------------------
# 📊 nDCG and MRR Calculation
# ----------------------------
def dcg_score(labels):
    return sum([(2 ** label - 1) / np.log2(i + 2) for i, label in enumerate(labels)])

def mrr_score(labels):
    # Find position of first relevant item (label=1)
    for idx, label in enumerate(labels):
        if label == 1:
            return 1.0 / (idx + 1)
    return 0.0  # No relevant items found

def ndcg_k(r, k):
    r = np.asarray(r)[:k]
    ideal = sorted(r, reverse=True)
    return dcg_score(r) / dcg_score(ideal) if dcg_score(ideal) != 0 else 0

def evaluate_metrics(model, df, k=10):
    model.eval()
    user_groups = df.groupby('userIdx')
    total_ndcg = 0
    total_mrr = 0
    total_users = 0

    for uid, group in tqdm(user_groups):
        items = torch.tensor(group.itemIdx.values, dtype=torch.long)
        users = torch.tensor([uid]*len(items), dtype=torch.long)
        labels = group.label.values
        with torch.no_grad():
            scores = torch.sigmoid(model(users, items)).cpu().numpy()
        
        # Sort labels according to scores (descending order)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_labels = labels[sorted_indices]
        
        # Calculate nDCG@k
        total_ndcg += ndcg_k(sorted_labels, k)
        
        # Calculate MRR
        total_mrr += mrr_score(sorted_labels)
        
        total_users += 1
    
    print(f"nDCG@{k}: {total_ndcg / total_users:.4f}")
    print(f"MRR: {total_mrr / total_users:.4f}")
    
    return total_ndcg / total_users, total_mrr / total_users

# ----------------------------
# 📈 Run Evaluation
# ----------------------------
evaluate_model(model, test_loader)
# Run evaluations with both nDCG and MRR
evaluate_metrics(model, test_behaviors, k=5)
evaluate_metrics(model, test_behaviors, k=10)