import torch
import torch.nn as nn
import torch.nn.functional as F

def spectral_representation(features, matrix_adj_eigenVectors):

    projected = torch.matmul(matrix_adj_eigenVectors, features)
    spectrum = torch.sum(projected, dim=2)

    return spectrum

def loss_spectrum(features, matrix_0utlier_ratios, matrix_adj, matrix_adj_eigenValues, matrix_adj_eigenVectors, matrix_average_distances, matrix_neighbors, batchsize):

    feature1 = features.unsqueeze(1)
    feature2 = features.unsqueeze(0)

    node_distances = torch.norm(feature1 - feature2, dim=-1)

    avg_dist1 = matrix_average_distances.unsqueeze(1)
    avg_dist2 = matrix_average_distances.unsqueeze(0)
    outlier_ratio1 = matrix_0utlier_ratios.unsqueeze(1)
    outlier_ratio2 = matrix_0utlier_ratios.unsqueeze(0)

    subgraph_distances = torch.zeros(batchsize, batchsize, 128, requires_grad=False).to(outlier_ratio2.device)

    for i in range(batchsize):
        for j in range(batchsize):
            for k in range(128):
                neighbors_i = matrix_neighbors[i]
                neighbors_j = matrix_neighbors[j]

                valid_neighbors_i = neighbors_i[k][neighbors_i[k] < 128]
                valid_neighbors_j = neighbors_j[k][neighbors_j[k] < 128]

                neighbors_i_valid = features[i, valid_neighbors_i]
                neighbors_j_valid = features[j, valid_neighbors_j]

                if neighbors_i_valid.size(0) > 0 and neighbors_j_valid.size(0) > 0:

                    inner_product = torch.matmul(neighbors_i_valid, neighbors_j_valid.T)
                    subgraph_distances[i, j, k] = torch.abs(inner_product).sum()
                    # subgraph_distances[j, i] = torch.abs(inner_product).sum(dim=1)

    total_spatial_distance = torch.log(
        (torch.abs(avg_dist1 - avg_dist2) + subgraph_distances + torch.abs(node_distances)) * (outlier_ratio1 + outlier_ratio2) + 1
    )

    spectrum = spectral_representation(features, matrix_adj_eigenVectors)
    # matrix_adj = torch.eye(128).repeat(features.size(0), 1, 1)
    matrix_norm_eigenValues = torch.nn.functional.softmax(matrix_adj_eigenValues, dim=1)
    kl_matrix = torch.zeros((batchsize, batchsize), requires_grad=False).to(matrix_norm_eigenValues.device)

    for i in range(batchsize):
        for j in range(i + 1, batchsize):
            kl_div_ij = F.kl_div(matrix_norm_eigenValues[i].log(), matrix_norm_eigenValues[j], reduction='batchmean')
            kl_matrix[i, j] = kl_div_ij
            kl_matrix[j, i] = kl_div_ij
    diff = spectrum.unsqueeze(1) - spectrum.unsqueeze(0)
    distance_Lambda_matrix = diff.abs()
    beta = 0.5
    total_spectral_distance = distance_Lambda_matrix + beta * kl_matrix.unsqueeze(-1)

    # classification stage
    return total_spatial_distance, total_spectral_distance

def loss_spectrum_nons(features, matrix_0utlier_ratios, matrix_adj, matrix_adj_eigenValues, matrix_adj_eigenVectors, matrix_average_distances, matrix_neighbors, batchsize):

    feature1 = features.unsqueeze(1)
    feature2 = features.unsqueeze(0)

    node_distances = torch.norm(feature1 - feature2, dim=-1)

    avg_dist1 = matrix_average_distances.unsqueeze(1)
    avg_dist2 = matrix_average_distances.unsqueeze(0)
    outlier_ratio1 = matrix_0utlier_ratios.unsqueeze(1)
    outlier_ratio2 = matrix_0utlier_ratios.unsqueeze(0)

    subgraph_distances = torch.zeros(batchsize, batchsize, 128, requires_grad=False).to(outlier_ratio2.device) # shape: (10, 10, 128)

    total_spatial_distance = torch.log(
        (torch.abs(avg_dist1 - avg_dist2) + torch.abs(node_distances)) * (outlier_ratio1 + outlier_ratio2) + 1
    )
    # print(total_spatial_distance.shape)

    spectrum = spectral_representation(features, matrix_adj_eigenVectors)
    # matrix_adj = torch.eye(128).repeat(features.size(0), 1, 1)
    matrix_norm_eigenValues = torch.nn.functional.softmax(matrix_adj_eigenValues, dim=1)
    kl_matrix = torch.zeros((batchsize, batchsize), requires_grad=False).to(matrix_norm_eigenValues.device)

    kl_matrix = F.kl_div(matrix_norm_eigenValues.unsqueeze(1).log(), matrix_norm_eigenValues.unsqueeze(0), reduction='none')
    kl_matrix = kl_matrix.sum(dim=2)
    diff = spectrum.unsqueeze(1) - spectrum.unsqueeze(0)
    distance_Lambda_matrix = diff.abs()
    beta = 0.5

    total_spectral_distance = distance_Lambda_matrix + beta * kl_matrix.unsqueeze(-1)

    # classification stage
    return total_spatial_distance, total_spectral_distance

def loss_spectrum_random(features, matrix_0utlier_ratios, matrix_adj, matrix_adj_eigenValues, matrix_adj_eigenVectors, matrix_average_distances, matrix_neighbors, batchsize):

    node_distances = torch.norm(features[:features.size(0) // 2] - features[features.size(0) // 2:], dim=-1)

    total_spatial_distance = torch.log(
        (torch.abs(matrix_average_distances[:features.size(0) // 2] - matrix_average_distances[features.size(0) // 2:]) +
         torch.abs(node_distances)) * (matrix_0utlier_ratios[:features.size(0) // 2] + matrix_0utlier_ratios[features.size(0) // 2:]) + 1
    )

    spectrum = spectral_representation(features, matrix_adj_eigenVectors)
    # matrix_adj = torch.eye(128).repeat(features.size(0), 1, 1)
    matrix_norm_eigenValues = torch.nn.functional.softmax(matrix_adj_eigenValues, dim=1)
    kl_matrix = torch.zeros((batchsize, batchsize), requires_grad=False).to(matrix_norm_eigenValues.device)
    kl_matrix = F.kl_div(matrix_norm_eigenValues[:features.size(0) // 2].log(), matrix_norm_eigenValues[features.size(0) // 2:], reduction='none')
    kl_matrix = kl_matrix.sum(dim=1)
    diff = spectrum[:features.size(0) // 2] - spectrum[features.size(0) // 2:]  # (batch, batch, 128)
    distance_Lambda_matrix = diff.abs()  # (batch, batch, 128)
    beta = 0.5
    total_spectral_distance = distance_Lambda_matrix + beta * kl_matrix.unsqueeze(-1)

    return total_spatial_distance, total_spectral_distance

class SSOGD(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, batch_size):
        super(SSOGD, self).__init__()
        self.GCN_l1 = nn.Linear(in_feats, hidden_feats)
        self.GCN_l2 = nn.Linear(hidden_feats, hidden_feats)
        self.GCN_l3 = nn.Linear(hidden_feats, int(hidden_feats/2))
        self.linear_att = nn.Linear(batch_size * 128, batch_size * 128)
        self.GCN_l4 = nn.Linear(int(hidden_feats/2), int(hidden_feats/2))
        self.linear4 = nn.Linear(int(128 * hidden_feats/2), int(hidden_feats/2))
        self.linear5 = nn.Linear(int(hidden_feats/2), out_feats)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(hidden_feats)

    def forward(self, features, matrix_0utlier_ratios, matrix_adj_eigenValues, adj,
                matrix_adj_eigenVectors, matrix_average_distances, matrix_neighbors, batch_size):
        x = torch.matmul(adj, features)
        x = self.GCN_l1(x)
        x = torch.matmul(adj, x)
        x = self.GCN_l2(x)
        # x = x.transpose(1, 2)
        # x = self.bn1(x)
        # x = x.transpose(1, 2)
        x = torch.matmul(adj, x)
        x = F.leaky_relu(self.GCN_l3(x), negative_slope=0.05)
        x = self.dropout1(x)
        if self.training:
            total_spatial_distance, total_spectral_distance = loss_spectrum_random(x, matrix_0utlier_ratios, adj,
                                                                                matrix_adj_eigenValues, matrix_adj_eigenVectors,
                                                                                matrix_average_distances, matrix_neighbors,
                                                                                batch_size)
            flattened_total_spatial_distance = torch.flatten(total_spatial_distance)
            flattened_total_spectral_distance = torch.flatten(total_spectral_distance)
            flattened_total_spectral_distance = self.linear_att(flattened_total_spectral_distance)
            flattened_total_spectral_distance = flattened_total_spectral_distance.abs()
            loss_gd = torch.dot(flattened_total_spatial_distance, flattened_total_spectral_distance)/flattened_total_spectral_distance.size(0)
            loss_gd = loss_gd/100
            loss_gd = loss_gd.abs()
        else:
            loss_gd = 0.1
        x = torch.matmul(adj, x)
        x = self.GCN_l4(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = self.linear4(x)
        x = self.linear5(x)
        return x, loss_gd

class GCNbaseline(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, batch_size):
        super(GCNbaseline, self).__init__()
        self.GCN_l1 = nn.Linear(in_feats, hidden_feats)
        self.GCN_l2 = nn.Linear(hidden_feats, hidden_feats)
        self.GCN_l3 = nn.Linear(hidden_feats, hidden_feats)
        self.linear2 = nn.Linear(128 * hidden_feats, hidden_feats)
        self.linear3 = nn.Linear(hidden_feats, out_feats)
        self.dropout1 = nn.Dropout(p=0.2)

    def forward(self, features, matrix_0utlier_ratios, matrix_adj_eigenValues, adj,
                matrix_adj_eigenVectors, matrix_average_distances, matrix_neighbors, batch_size):
        x = torch.matmul(adj, features)
        x = self.GCN_l1(x)
        x = torch.matmul(adj, x)
        x = self.GCN_l2(x)
        x = torch.matmul(adj, x)
        x = F.leaky_relu(self.GCN_l3(x), negative_slope=0.05)
        x = self.dropout1(x)
        x = x.view(batch_size, -1)
        loss_gd = 1.0
        x = self.linear2(x)
        x = self.linear3(x)
        return x, loss_gd
