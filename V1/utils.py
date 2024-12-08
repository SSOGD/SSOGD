import numpy as np
import torch
import h5py

def load_citation(subject=None, session=None):

    trial_num = 1
    idx_train_and_val= list(range(0, subject*int(7680/trial_num/6)))
    idx_train_and_val.extend(list(range(subject*int(7680/trial_num/6)+int(7680/trial_num/6), int(7680/trial_num))))
    idx_train = idx_train_and_val
    # idx_val = idx_train_and_val[int(4 * 7680 / trial_num / 6):]
    idx_test = range(subject*int(7680/trial_num/6), subject*int(7680/trial_num/6)+int(7680/trial_num/6))
    # porting to pytorch
    file_path = '/home/anaconda3/envs/firsttorch/DATA/matlab_save_7680_sumrhythm_003.mat'
    with h5py.File(file_path, 'r') as f:
        data = {key: f[key][:] for key in f.keys()}
    matrix_labels = np.transpose(np.array(data['matrix_labels']), (1, 0))
    matrix_features = np.transpose(np.array(data['matrix_features']), (3, 2, 1, 0))
    matrix_0utlier_ratios = np.transpose(np.array(data['matrix_0utlier_ratios']), (1, 0))
    matrix_adj_eigenValues = np.transpose(np.array(data['matrix_adj_eigenValues']), (1, 0))
    matrix_adj = np.transpose(np.array(data['matrix_adj']), (2, 1, 0))
    matrix_adj_eigenVectors = np.transpose(np.array(data['matrix_adj_eigenVectors']), (2, 1, 0))
    matrix_average_distances = np.transpose(np.array(data['matrix_average_distances']), (1, 0))
    matrix_neighbors = np.transpose(np.array(data['matrix_neighbors']), (2, 1, 0))

    # features = torch.FloatTensor(np.array(data['data_tl_dab_r3'][session]).reshape(int(7680/trial_num), 256)).float()
    labels = torch.LongTensor(matrix_labels.reshape(int(7680/trial_num)))
    features = torch.FloatTensor(matrix_features[session]).float()
    matrix_0utlier_ratios = torch.FloatTensor(matrix_0utlier_ratios).float()
    matrix_adj_eigenValues = torch.FloatTensor(matrix_adj_eigenValues).float()
    matrix_adj = torch.FloatTensor(matrix_adj).float()
    diagonal_addition = torch.eye(128).unsqueeze(0)
    matrix_adj = diagonal_addition + matrix_adj
    matrix_adj_eigenVectors = torch.FloatTensor(matrix_adj_eigenVectors).float()
    matrix_average_distances = torch.FloatTensor(matrix_average_distances).float()
    matrix_neighbors = torch.LongTensor(matrix_neighbors)

    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    # idx_val = torch.LongTensor(idx_val)

    return (labels, features, matrix_0utlier_ratios,
            matrix_adj_eigenValues, matrix_adj,
            matrix_adj_eigenVectors, matrix_average_distances,
            matrix_neighbors, idx_train, idx_test)

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)