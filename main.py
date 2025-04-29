import numpy as np
import torch
import logging
from tqdm.auto import tqdm
import torch.optim as optim
import os
from pathlib import Path
import random
from torch.utils.data import DataLoader
import importlib
import subprocess
from sklearn.metrics import roc_auc_score
from utils import mrr_score, ndcg_score

import utils
from parameters import parse_args
from preprocess import read_news, get_doc_input
from prepare_data import prepare_training_data, prepare_testing_data
from dataset import DatasetTrain, DatasetTest, NewsDataset
import nltk
from torch.nn.utils.rnn import pad_sequence
import torch

def collate_custom_fn(batch):
    log_vecs, log_masks, news_vecs, labels = zip(*batch)
    
    # Convert to PyTorch tensors if they're not already
    log_vecs = [torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in log_vecs]
    log_masks = [torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in log_masks]
    news_vecs = [torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in news_vecs]
    labels = [torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in labels]

    # Pad sequences
    log_vecs_padded = pad_sequence(log_vecs, batch_first=True)  # shape: [B, max_seq_len, 400]
    log_masks_padded = pad_sequence(log_masks, batch_first=True)  # [B, max_seq_len]
    news_vecs_stacked = pad_sequence(news_vecs, batch_first=True)  # Assuming these are fixed-size

    return log_vecs_padded, log_masks_padded, news_vecs_stacked, labels

def train(rank, args):
    # Set rank and distributed settings to False for single machine usage
    is_distributed = False
    rank = 0

    # Initialize logger and set the device to CPU
    utils.setuplogger()
    device = torch.device('cpu')  # Use CPU

    # Read and preprocess the data
    news, news_index, category_dict, subcategory_dict, word_dict = read_news(
        os.path.join(args.train_data_dir, 'news.tsv'), args, mode='train')

    news_title, news_category, news_subcategory = get_doc_input(
        news, news_index, category_dict, subcategory_dict, word_dict, args)
    news_combined = np.concatenate([x for x in [news_title, news_category, news_subcategory] if x is not None], axis=-1)

    if rank == 0:
        logging.info('Initializing word embedding matrix...')

    embedding_matrix, have_word = utils.load_matrix(args.glove_embedding_path,
                                                    word_dict,
                                                    args.word_embedding_dim)
    if rank == 0:
        logging.info(f'Word dict length: {len(word_dict)}')
        logging.info(f'Have words: {len(have_word)}')
        logging.info(f'Missing rate: {(len(word_dict) - len(have_word)) / len(word_dict)}')

    module = importlib.import_module(f'model.{args.model}')
    if args.model== 'NRMS':
        model = module.Model(args, embedding_matrix)
    else:
        model = module.Model(args, embedding_matrix, len(category_dict), len(subcategory_dict))

    #model = module.Model(args, embedding_matrix, len(category_dict), len(subcategory_dict))

    if args.load_ckpt_name is not None:
        ckpt_path = utils.get_checkpoint(args.model_dir, args.load_ckpt_name)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Model loaded from {ckpt_path}.")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Move model to CPU
    model = model.to(device)

    # Load dataset and dataloader
    data_file_path = os.path.join(args.train_data_dir, f'behaviors_np{args.npratio}_{rank}.tsv')
    dataset = DatasetTrain(data_file_path, news_index, news_combined, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    logging.info('Training...')
    for ep in range(args.start_epoch, args.epochs):
        loss = 0.0
        accuary = 0.0
        for cnt, (log_ids, log_mask, input_ids, targets) in enumerate(dataloader):
            log_ids = log_ids.to(device, non_blocking=True)
            log_mask = log_mask.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            bz_loss, y_hat = model(log_ids, log_mask, input_ids, targets)
            loss += bz_loss.data.float()
            accuary += utils.acc(targets, y_hat)
            optimizer.zero_grad()
            bz_loss.backward()
            optimizer.step()

            if cnt % args.log_steps == 0:
                logging.info(
                    '[{}] Ed: {}, train_loss: {:.5f}, acc: {:.5f}'.format(
                        rank, cnt * args.batch_size, loss.data / cnt, accuary / cnt)
                )

            if rank == 0 and cnt != 0 and cnt % args.save_steps == 0:
                ckpt_path = os.path.join(args.model_dir, f'epoch-{ep+1}-{cnt}.pt')
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'category_dict': category_dict,
                        'word_dict': word_dict,
                        'subcategory_dict': subcategory_dict
                    }, ckpt_path)
                logging.info(f"Model saved to {ckpt_path}.")

        logging.info('Training finish.')

        if rank == 0:
            ckpt_path = os.path.join(args.model_dir, f'epoch-{ep+1}.pt')
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'category_dict': category_dict,
                    'subcategory_dict': subcategory_dict,
                    'word_dict': word_dict,
                }, ckpt_path)
            logging.info(f"Model saved to {ckpt_path}.")


def test(rank, args):
    # Similar adjustments for testing mode
    is_distributed = False
    rank = 0

    utils.setuplogger()
    device = torch.device('cpu')  # Use CPU

    if args.load_ckpt_name is not None:
        ckpt_path = utils.get_checkpoint(args.model_dir, args.load_ckpt_name)

    assert ckpt_path is not None, 'No checkpoint found.'
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    subcategory_dict = checkpoint['subcategory_dict']
    category_dict = checkpoint['category_dict']
    word_dict = checkpoint['word_dict']

    dummy_embedding_matrix = np.zeros((len(word_dict) + 1, args.word_embedding_dim))
    module = importlib.import_module(f'model.{args.model}')
    model = module.Model(args, dummy_embedding_matrix, len(category_dict), len(subcategory_dict))
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Model loaded from {ckpt_path}")

    model = model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    # Testing data processing remains the same
    news, news_index = read_news(os.path.join(args.test_data_dir, 'news.tsv'), args, mode='test')
    news_title, news_category, news_subcategory = get_doc_input(
        news, news_index, category_dict, subcategory_dict, word_dict, args)
    news_combined = np.concatenate([x for x in [news_title, news_category, news_subcategory] if x is not None], axis=-1)

    news_dataset = NewsDataset(news_combined)
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=4)

    # Evaluating news scoring on CPU
    news_scoring = []
    with torch.no_grad():
        for input_ids in tqdm(news_dataloader):
            input_ids = input_ids.to(device)
            news_vec = model.news_encoder(input_ids)
            news_vec = news_vec.to(torch.device("cpu")).detach().numpy()
            news_scoring.extend(news_vec)

    news_scoring = np.array(news_scoring)
    logging.info("news scoring num: {}".format(news_scoring.shape[0]))
    data_file_path = os.path.join(args.test_data_dir, f'behaviors_{rank}.tsv')
    # Collate function and testing loop
    dataset = DatasetTest(data_file_path, news_index, news_scoring, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,collate_fn=collate_custom_fn)

    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 = []

    for cnt, (log_vecs, log_mask, news_vecs, labels) in enumerate(dataloader):
        log_vecs = log_vecs.to(device)
        log_mask = log_mask.to(device)

        user_vecs = model.user_encoder(log_vecs, log_mask).to(torch.device("cpu")).detach().numpy()

        for user_vec, news_vec, label in zip(user_vecs, news_vecs, labels):
            label = label.float()
            if label.mean().item() == 0. or label.mean().item() == 1.:
                continue

            label = label.detach().cpu().numpy()
            valid_news_vec = news_vec[:len(label)]  # this is critical
            score = np.dot(valid_news_vec, user_vec)
            # print(score)
            # print(label)
            # print(label.sum())
            # Compute metrics
            auc = roc_auc_score(label, score)
            mrr = mrr_score(label, score)
            ndcg5 = ndcg_score(label, score, k=5)
            ndcg10 = ndcg_score(label, score, k=10)

            AUC.append(auc)
            MRR.append(mrr)
            nDCG5.append(ndcg5)
            nDCG10.append(ndcg10)

        if cnt % args.log_steps == 0:
            logging.info("[{}] {} samples: AUC: {:.6f}, MRR: {:.6f}, nDCG5: {:.6f}, nDCG10: {:.6f}".format(
                rank, cnt * args.batch_size, np.mean(AUC), np.mean(MRR), np.mean(nDCG5), np.mean(nDCG10)))

    logging.info('Testing finished.')

if __name__ == "__main__":
    utils.setuplogger()
    args = parse_args()
    utils.dump_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    if 'train' in args.mode:
        if args.prepare:
            logging.info('Preparing training data...')
            total_sample_num = prepare_training_data(args.train_data_dir, args.nGPU, args.npratio, args.seed)
        else:
            total_sample_num = 0
            for i in range(args.nGPU):
                data_file_path = os.path.join(args.train_data_dir, f'behaviors_np{args.npratio}_{i}.tsv')
                if not os.path.exists(data_file_path):
                    logging.error(f'Splited training data {data_file_path} for GPU {i} does not exist. Please set the parameter --prepare as True and rerun the code.')
                    exit()
                result = subprocess.getoutput(f'wc -l {data_file_path}')
                total_sample_num += int(result.split(' ')[0])
            logging.info('Skip training data preparation.')

        train(None, args)


    if 'test' in args.mode:
        if args.prepare:
            logging.info('Preparing testing data...')
            total_sample_num = prepare_testing_data(args.test_data_dir, args.nGPU)
        else:
            total_sample_num = 0
            for i in range(args.nGPU):
                data_file_path = os.path.join(args.test_data_dir, f'behaviors_{i}.tsv')
                if not os.path.exists(data_file_path):
                    logging.error(f'Splited testing data {data_file_path} for GPU {i} does not exist. Please set the parameter --prepare as True and rerun the code.')
                    exit()

        test(None, args)
