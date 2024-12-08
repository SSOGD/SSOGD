import argparse
import numpy as np
import torch.optim as optim
from time import perf_counter
from utils import load_citation, set_seed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dgc import SSOGD
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="cora",
                    help='Dataset to use.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--gdlosswight', type=float, default=0.1,
                    help='gdlosswight.')
parser.add_argument('--batch_size', type=int, default=160,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden_feats', type=int, default=64,
                    help='Weight parameters.')
parser.add_argument('--out_feats', type=int, default=2,
                    help='Weight parameters.')
parser.add_argument('--device', default='cuda:7',
                    help='device to use for training / testing')
parser.add_argument('--num_epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--loss_file', type=str, default='tt_right_loss_7680_6fold_aumrhythm_001_L5_noBN_lr0.001yc64nonsgdlossw0.1.txt',
                    help='Path to the file where loss values are stored.')

args = parser.parse_args()

# 定义数据集
class GraphDataset(Dataset):
    def __init__(self, labels_train, features_train, matrix_0utlier_ratios_train, matrix_adj_eigenValues_train,
                           matrix_adj_train, matrix_adj_eigenVectors_train, matrix_average_distances_train, matrix_neighbors_train):
        self.labels_train = labels_train
        self.features_train = features_train
        self.matrix_0utlier_ratios_train = matrix_0utlier_ratios_train
        self.matrix_adj_eigenValues_train = matrix_adj_eigenValues_train
        self.matrix_adj_train = matrix_adj_train
        self.matrix_adj_eigenVectors_train = matrix_adj_eigenVectors_train
        self.matrix_average_distances_train = matrix_average_distances_train
        self.matrix_neighbors_train = matrix_neighbors_train

    def __len__(self):
        return len(self.labels_train)

    def __getitem__(self, idx):
        labels_train = self.labels_train[idx]
        features_train = self.features_train[idx]
        matrix_0utlier_ratios_train = self.matrix_0utlier_ratios_train[idx]
        matrix_adj_eigenValues_train = self.matrix_adj_eigenValues_train[idx]
        matrix_adj_train = self.matrix_adj_train[idx]
        matrix_adj_eigenVectors_train = self.matrix_adj_eigenVectors_train[idx]
        matrix_average_distances_train = self.matrix_average_distances_train[idx]
        matrix_neighbors_train = self.matrix_neighbors_train[idx]
        return labels_train, features_train, matrix_0utlier_ratios_train, matrix_adj_eigenValues_train, matrix_adj_train, matrix_adj_eigenVectors_train, matrix_average_distances_train, matrix_neighbors_train

def evaluate_model(model, dataloader_test, device, batch_size):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for (labels_test_batch, features_test_batch, matrix_0utlier_ratios_test_batch,
             matrix_adj_eigenValues_test_batch, matrix_adj_test_batch, 
             matrix_adj_eigenVectors_test_batch, matrix_average_distances_test_batch,
             matrix_neighbors_test_batch) in dataloader_test:

            labels_test_batch = labels_test_batch.to(device)
            features_test_batch = features_test_batch.to(device)
            matrix_0utlier_ratios_test_batch = matrix_0utlier_ratios_test_batch.to(device)
            matrix_adj_eigenValues_test_batch = matrix_adj_eigenValues_test_batch.to(device)
            matrix_adj_test_batch = matrix_adj_test_batch.to(device)
            matrix_adj_eigenVectors_test_batch = matrix_adj_eigenVectors_test_batch.to(device)
            matrix_average_distances_test_batch = matrix_average_distances_test_batch.to(device)
            matrix_neighbors_test_batch = matrix_neighbors_test_batch.to(device)

            test_outputs, loss_gd = model(features_test_batch, matrix_0utlier_ratios_test_batch,
                                          matrix_adj_eigenValues_test_batch, matrix_adj_test_batch,
                                          matrix_adj_eigenVectors_test_batch,
                                          matrix_average_distances_test_batch, matrix_neighbors_test_batch,
                                          batch_size)

            predicted_labels = torch.argmax(test_outputs, dim=1)

            total += labels_test_batch.size(0)
            correct += (predicted_labels == labels_test_batch).sum().item()

    accuracy = correct / total
    return accuracy




def main_SSOGD(args, subject, session):
    # load data
    labels, features, matrix_0utlier_ratios, matrix_adj_eigenValues, matrix_adj, matrix_adj_eigenVectors, matrix_average_distances, matrix_neighbors, idx_train, idx_test = load_citation(subject, session)
    batch_size = args.batch_size

    model = SSOGD(features.size(2), args.hidden_feats, args.out_feats, batch_size)
    model = model.to(args.device)

    labels_train = labels[idx_train]
    features_train = features[idx_train]
    matrix_0utlier_ratios_train = matrix_0utlier_ratios[idx_train]
    matrix_adj_eigenValues_train = matrix_adj_eigenValues[idx_train]
    matrix_adj_train = matrix_adj[idx_train]
    matrix_adj_eigenVectors_train = matrix_adj_eigenVectors[idx_train]
    matrix_average_distances_train = matrix_average_distances[idx_train]
    matrix_neighbors_train = matrix_neighbors[idx_train]

    labels_test = labels[idx_test]
    features_test = features[idx_test]
    matrix_0utlier_ratios_test = matrix_0utlier_ratios[idx_test]
    matrix_adj_eigenValues_test = matrix_adj_eigenValues[idx_test]
    matrix_adj_test = matrix_adj[idx_test]
    matrix_adj_eigenVectors_test = matrix_adj_eigenVectors[idx_test]
    matrix_average_distances_test = matrix_average_distances[idx_test]
    matrix_neighbors_test = matrix_neighbors[idx_test]

    dataset = GraphDataset(labels_train, features_train, matrix_0utlier_ratios_train, matrix_adj_eigenValues_train,
                           matrix_adj_train, matrix_adj_eigenVectors_train, matrix_average_distances_train, matrix_neighbors_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    dataset_test = GraphDataset(labels_test, features_test, matrix_0utlier_ratios_test, matrix_adj_eigenValues_test,
                           matrix_adj_test, matrix_adj_eigenVectors_test, matrix_average_distances_test, matrix_neighbors_test)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, drop_last=True)

    batches = list(dataloader_test)
    # dataset_val = GraphDataset(labels_val, features_val, matrix_0utlier_ratios_val, matrix_adj_eigenValues_val,
    #                        matrix_adj_val, matrix_adj_eigenVectors_val, matrix_average_distances_val, matrix_neighbors_val)
    # dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # accuracy_val_list = []
    accuracy_test_list = []

    # 训练循环
    num_epochs = args.num_epochs
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_loss_gd = 0
        batch_index = 0
        total_loss_val = 0
        total_loss_gd_val = 0
        batch_index_val = 0
        for (labels_train_batch_trans, features_train_batch_trans, matrix_0utlier_ratios_train_batch_trans, matrix_adj_eigenValues_train_batch_trans,
             matrix_adj_train_batch_trans, matrix_adj_eigenVectors_train_batch_trans, matrix_average_distances_train_batch_trans, matrix_neighbors_train_batch_trans) in dataloader:

            random_batch = random.choice(batches)
            (labels_tt_batch, features_tt_batch, matrix_0utlier_ratios_tt_batch, matrix_adj_eigenValues_tt_batch,
             matrix_adj_tt_batch, matrix_adj_eigenVectors_tt_batch, matrix_average_distances_tt_batch, matrix_neighbors_tt_batch) = random_batch

            features_train_batch = torch.cat((features_train_batch_trans, features_tt_batch), dim=0)
            matrix_0utlier_ratios_train_batch = torch.cat((matrix_0utlier_ratios_train_batch_trans, matrix_0utlier_ratios_tt_batch), dim=0)
            matrix_adj_eigenValues_train_batch = torch.cat((matrix_adj_eigenValues_train_batch_trans, matrix_adj_eigenValues_tt_batch), dim=0)
            matrix_adj_train_batch = torch.cat((matrix_adj_train_batch_trans, matrix_adj_tt_batch), dim=0)
            matrix_adj_eigenVectors_train_batch = torch.cat((matrix_adj_eigenVectors_train_batch_trans, matrix_adj_eigenVectors_tt_batch), dim=0)
            matrix_average_distances_train_batch = torch.cat((matrix_average_distances_train_batch_trans, matrix_average_distances_tt_batch), dim=0)
            matrix_neighbors_train_batch = torch.cat((matrix_neighbors_train_batch_trans, matrix_neighbors_tt_batch), dim=0)
            labels_train_batch = labels_train_batch_trans.to(args.device)
            features_train_batch = features_train_batch.to(args.device)
            matrix_0utlier_ratios_train_batch = matrix_0utlier_ratios_train_batch.to(args.device)
            matrix_adj_eigenValues_train_batch = matrix_adj_eigenValues_train_batch.to(args.device)
            matrix_adj_train_batch = matrix_adj_train_batch.to(args.device)
            matrix_adj_eigenVectors_train_batch = matrix_adj_eigenVectors_train_batch.to(args.device)
            matrix_average_distances_train_batch = matrix_average_distances_train_batch.to(args.device)
            matrix_neighbors_train_batch = matrix_neighbors_train_batch.to(args.device)

            optimizer.zero_grad()
            outputs, loss_gd = model(features_train_batch, matrix_0utlier_ratios_train_batch,
                                     matrix_adj_eigenValues_train_batch, matrix_adj_train_batch, matrix_adj_eigenVectors_train_batch,
                                     matrix_average_distances_train_batch, matrix_neighbors_train_batch, batch_size)
            loss_cls = criterion(outputs[:len(outputs) // 2], labels_train_batch)
            loss = loss_cls + args.gdlosswight * loss_gd
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_loss_gd += loss_gd.item()
            # print(f'Iteration [{batch_index + 1}/{len(dataloader)}]')
            batch_index = batch_index + 1

        avg_loss = total_loss / len(dataloader)
        avg_loss_gd = total_loss_gd / len(dataloader)
        # avg_loss_val = total_loss_val / len(dataloader_val)
        # accuracy_val = evaluate_model(model, dataloader_val, args.device, args.batch_size)
        accuracy_test = evaluate_model(model, dataloader_test, args.device, args.batch_size)
        # accuracy_train = evaluate_model(model, dataloader, args.device, args.batch_size)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Loss_gd: {avg_loss_gd:.4f}, '
              f'Accuracy_test: {accuracy_test * 100:.2f}%')

        a_l = open(args.loss_file, 'a')
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Loss_gd: {avg_loss_gd:.4f}, '
            f' Accuracy_test: {accuracy_test * 100:.2f}%', file=a_l)
        a_l.close()
        accuracy_test_list.append(accuracy_test)

    result_list_rank = []
    for n in range(5, len(accuracy_test_list) + 1, 5):
        result_list_rank.append(accuracy_test_list[n-1])
    result_tenser_rank = torch.tensor(result_list_rank)
    return result_tenser_rank

for i in range(5):
    result_tenser_rank_list = []
    for j in range(6):
        result_tenser_rank = main_SSOGD(args, subject=j, session=i)
        result_tenser_rank_list.append(result_tenser_rank)
    stacked_tensor_rank = torch.stack(result_tenser_rank_list)

