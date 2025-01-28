import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import GradientBoostingRegressor  # 使用 XGBoost 的替代品
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dense_to_sparse
import logging
import joblib  # 用于保存和加载模型组件
import os
import json
import torch.nn.functional as F

# 设置日志
logging.basicConfig(
    level=logging.INFO,  # 日志级别
    format="%(asctime)s - %(name)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # 输出到控制台
)

logger = logging.getLogger(__name__)

# """ 普通GAT 无Res"""
# class GATEncoder(nn.Module):
#     def __init__(self, in_channels, encoder_hidden_channels, latent_dim, num_layers=3, heads=1, dropout_prob=0.01):
#         super(GATEncoder, self).__init__()
#         self.linear_input = nn.Linear(in_channels, encoder_hidden_channels * heads)
#         self.gat_layers = nn.ModuleList()
#         for _ in range(num_layers):
#             self.gat_layers.append(GATConv(encoder_hidden_channels * heads, encoder_hidden_channels, heads=heads, concat=True))
#         self.latent_layer = GATConv(encoder_hidden_channels * heads, latent_dim, heads=1, concat=False)
#         self.dropout = nn.Dropout(p=dropout_prob)
#         self.leaky_relu = nn.LeakyReLU(0.2)

#     def forward(self, x, edge_index):
#         h = self.linear_input(x)
#         for layer in self.gat_layers:
#             h = self.leaky_relu(layer(h, edge_index))
#             h = self.dropout(h)
#         latent = self.latent_layer(h, edge_index)
#         return latent
# class GATDecoder(nn.Module):
#     def __init__(self, latent_dim, decoder_hidden_channels, out_channels, num_layers=3, heads=1, dropout_prob=0.01):
#         super(GATDecoder, self).__init__()
#         self.linear_input = nn.Linear(latent_dim, decoder_hidden_channels * heads)
#         self.gat_layers = nn.ModuleList()
#         for _ in range(num_layers):
#             self.gat_layers.append(GATConv(decoder_hidden_channels * heads, decoder_hidden_channels, heads=heads, concat=True))
#         self.final_layer = nn.Linear(decoder_hidden_channels * heads, out_channels)
#         self.dropout = nn.Dropout(p=dropout_prob)

#     def forward(self, z, edge_index):
#         h = self.linear_input(z)
#         for layer in self.gat_layers:
#             h = F.leaky_relu(layer(h, edge_index), negative_slope=0.2)
#             h = self.dropout(h)
#         x_recon = self.final_layer(h)
#         return x_recon


# ''' SAGEConv '''
# class GATEncoder(nn.Module):
#     def __init__(self, in_channels, encoder_hidden_channels, latent_dim, num_layers=3, dropout_prob=0.01):
#         super(GATEncoder, self).__init__()
#         self.linear_input = nn.Linear(in_channels, encoder_hidden_channels)
#         self.sage_layers = nn.ModuleList()
#         self.linear_skip_layers = nn.ModuleList()

#         for _ in range(num_layers):
#             self.sage_layers.append(SAGEConv(encoder_hidden_channels, encoder_hidden_channels))
#             self.linear_skip_layers.append(nn.Linear(encoder_hidden_channels, latent_dim))

#         self.latent_layer = SAGEConv(encoder_hidden_channels, latent_dim)
#         self.dropout = nn.Dropout(p=dropout_prob)
#         self.leaky_relu = nn.LeakyReLU(0.2)

#     def forward(self, x, edge_index):
#         h = self.linear_input(x)
#         skip_connections = []

#         for layer in self.sage_layers:
#             skip_connections.append(h)
#             h = self.leaky_relu(layer(h, edge_index))
#             h = self.dropout(h)

#         latent = self.latent_layer(h, edge_index)

#         for i in range(len(skip_connections)):
#             skip_connections[i] = self.linear_skip_layers[i](skip_connections[i])

#         for skip in skip_connections:
#             latent = latent + skip
#         return latent


# class GATDecoder(nn.Module):
#     def __init__(self, latent_dim, decoder_hidden_channels, out_channels, num_layers=3, dropout_prob=0.01):
#         super(GATDecoder, self).__init__()

#         self.linear_input = nn.Linear(latent_dim, decoder_hidden_channels)
#         self.sage_layers = nn.ModuleList()
#         self.linear_skip_layers = nn.ModuleList()

#         for _ in range(num_layers):
#             self.sage_layers.append(SAGEConv(decoder_hidden_channels, decoder_hidden_channels))
#             self.linear_skip_layers.append(nn.Linear(decoder_hidden_channels, out_channels))

#         self.final_layer = nn.Linear(decoder_hidden_channels, out_channels)
#         self.dropout = nn.Dropout(p=dropout_prob)

#     def forward(self, z, edge_index):
#         h = self.linear_input(z)
#         skip_connections = []

#         for layer in self.sage_layers:
#             skip_connections.append(h)
#             h = F.leaky_relu(layer(h, edge_index), negative_slope=0.2)
#             h = self.dropout(h)

#         x_recon = self.final_layer(h)

#         for i in range(len(skip_connections)):
#             skip_connections[i] = self.linear_skip_layers[i](skip_connections[i])

#         for skip in skip_connections:
#             x_recon = x_recon + skip
#         return x_recon


''' GCN '''
class GATEncoder(nn.Module):
    def __init__(self, in_channels, encoder_hidden_channels, latent_dim, num_layers=3, dropout_prob=0.01):
        super(GATEncoder, self).__init__()
        # 对输入特征 x 进行线性变换，调整维度
        self.linear_input = nn.Linear(in_channels, encoder_hidden_channels)
        self.gcn_layers = nn.ModuleList()
        self.linear_skip_layers = nn.ModuleList()
        # 后续层GCN层：每层输入encoder_hidden_channels，输出encoder_hidden_channels
        for _ in range(num_layers):
            self.gcn_layers.append(GCNConv(encoder_hidden_channels, encoder_hidden_channels))
            self.linear_skip_layers.append(nn.Linear(encoder_hidden_channels, latent_dim))
        # Latent层：最后一层GCN，输出latent_dim
        self.latent_layer = GCNConv(encoder_hidden_channels, latent_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        # 先将输入特征 x 经过线性变换，调整维度为 encoder_hidden_channels
        h = self.linear_input(x)
        skip_connections = []

        # 对每一层应用GCN并记录跳跃连接
        for layer in self.gcn_layers:
            skip_connections.append(h)  # 记录当前层的输出（跳跃连接）
            h = self.leaky_relu(layer(h, edge_index))  # Apply GCNConv and LeakyReLU
            h = self.dropout(h)  # Dropout
        # 最后一层的latent表示
        latent = self.latent_layer(h, edge_index)
        # 使用线性层调整每个跳跃连接的维度，使其和latent一致
        for i in range(len(skip_connections)):
            skip_connections[i] = self.linear_skip_layers[i](skip_connections[i])
        # 将所有跳跃连接与latent加在一起
        for skip in skip_connections:
            latent = latent + skip
        return latent


class GATDecoder(nn.Module):
    def __init__(self, latent_dim, decoder_hidden_channels, out_channels, num_layers=3, dropout_prob=0.01):
        super(GATDecoder, self).__init__()

        self.linear_input = nn.Linear(latent_dim, decoder_hidden_channels)
        self.gcn_layers = nn.ModuleList()
        self.linear_skip_layers = nn.ModuleList()
        # GCN层：每层输入decoder_hidden_channels，输出decoder_hidden_channels
        for _ in range(num_layers):
            self.gcn_layers.append(GCNConv(decoder_hidden_channels, decoder_hidden_channels))
            self.linear_skip_layers.append(nn.Linear(decoder_hidden_channels, out_channels))  # 对应的线性层

        # 重构输出层，最后输出out_channels维度
        self.final_layer = nn.Linear(decoder_hidden_channels, out_channels)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, z, edge_index):
        h = self.linear_input(z)
        skip_connections = []

        # 对每一层应用GCN并记录跳跃连接
        for layer in self.gcn_layers:
            skip_connections.append(h)
            h = F.leaky_relu(layer(h, edge_index), negative_slope=0.2)
            h = self.dropout(h)  # Dropout
        # 最后一层的重构输出
        x_recon = self.final_layer(h)
        # 使用线性层调整每个跳跃连接的维度，使其和x_recon一致
        for i in range(len(skip_connections)):
            skip_connections[i] = self.linear_skip_layers[i](skip_connections[i])
        # 将所有跳跃连接与x_recon进行加法融合
        for skip in skip_connections:
            x_recon = x_recon + skip
        return x_recon


""" 原本GAT """
# class GATEncoder(nn.Module):
#     def __init__(self, in_channels, encoder_hidden_channels, latent_dim, num_layers=3, heads=1, dropout_prob=0.01):
#         super(GATEncoder, self).__init__()
#         # 对输入特征 x 进行线性变换，调整维度，使其与GATConv的输入维度一致
#         self.linear_input = nn.Linear(in_channels, encoder_hidden_channels * heads)
#         self.gat_layers = nn.ModuleList()
#         self.linear_skip_layers = nn.ModuleList()
#         # 后续层GAT层：每层输入encoder_hidden_channels * heads，输出encoder_hidden_channels * heads
#         # 为每一层的跳跃连接使用独立的线性变换
#         for _ in range(num_layers):
#             self.gat_layers.append(GATConv(encoder_hidden_channels * heads, encoder_hidden_channels, heads=heads, concat=True))
#             self.linear_skip_layers.append(nn.Linear(encoder_hidden_channels * heads, latent_dim))
#         # Latent层：最后一层GAT，输出latent_dim，注意这里concat=False
#         self.latent_layer = GATConv(encoder_hidden_channels * heads, latent_dim, heads=1, concat=False)
#         self.dropout = nn.Dropout(p=dropout_prob)
#         self.leaky_relu = nn.LeakyReLU(0.2)

#     def forward(self, x, edge_index):
#         # 先将输入特征 x 经过线性变换，调整维度为 encoder_hidden_channels * heads
#         h = self.linear_input(x)
#         skip_connections = []

#         # 对每一层应用GAT并记录跳跃连接
#         for layer in self.gat_layers:
#             skip_connections.append(h)  # 记录当前层的输出（跳跃连接）
#             h = self.leaky_relu(layer(h, edge_index))  # Apply GATConv and LeakyReLU
#             h = self.dropout(h)  # Dropout
#         # 最后一层的latent表示
#         latent = self.latent_layer(h, edge_index)
#         # 使用线性层调整每个跳跃连接的维度，使其和latent一致
#         for i in range(len(skip_connections)):
#             skip_connections[i] = self.linear_skip_layers[i](skip_connections[i])
#         # 将所有跳跃连接与latent加在一起
#         for skip in skip_connections:
#             latent = latent + skip
#         return latent
# class GATDecoder(nn.Module):
#     def __init__(self, latent_dim, decoder_hidden_channels, out_channels, num_layers=3, heads=1, dropout_prob=0.01):
#         super(GATDecoder, self).__init__()

#         self.linear_input = nn.Linear(latent_dim, decoder_hidden_channels * heads)
#         self.gat_layers = nn.ModuleList()
#         self.linear_skip_layers = nn.ModuleList()
#         # GAT层：每层输入decoder_hidden_channels * heads，输出decoder_hidden_channels * heads
#         for _ in range(num_layers):
#             self.gat_layers.append(GATConv(decoder_hidden_channels * heads, decoder_hidden_channels, heads=heads, concat=True))
#             self.linear_skip_layers.append(nn.Linear(decoder_hidden_channels * heads, out_channels))  # 对应的线性层

#         # 重构输出层，最后输出out_channels维度
#         self.final_layer = nn.Linear(decoder_hidden_channels * heads, out_channels)
#         self.dropout = nn.Dropout(p=dropout_prob)

#     def forward(self, z, edge_index):
#         h = self.linear_input(z)
#         skip_connections = []

#         # 对每一层应用GAT并记录跳跃连接
#         for layer in self.gat_layers:
#             skip_connections.append(h)
#             h = F.leaky_relu(layer(h, edge_index), negative_slope=0.2)
#             h = self.dropout(h)  # Dropout
#         # 最后一层的重构输出
#         x_recon = self.final_layer(h)
#         # 使用线性层调整每个跳跃连接的维度，使其和x_recon一致
#         for i in range(len(skip_connections)):
#             skip_connections[i] = self.linear_skip_layers[i](skip_connections[i])
#         # 将所有跳跃连接与x_recon进行加法融合
#         for skip in skip_connections:
#             x_recon = x_recon + skip
#         return x_recon


class Encoder(nn.Module):
    def __init__(self, in_channels, encoder_hidden_channels, latent_dim, num_layers=3, dropout_prob=0.01):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels, encoder_hidden_channels))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(encoder_hidden_channels, encoder_hidden_channels))

        self.latent_layer = nn.Linear(encoder_hidden_channels, latent_dim)

        self.dropout = nn.Dropout(p=dropout_prob)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = self.leaky_relu(layer(h))
            h = self.dropout(h)

        latent = self.latent_layer(h)
        return latent


class Decoder(nn.Module):
    def __init__(self, latent_dim, decoder_hidden_channels, out_channels, num_layers=3, dropout_prob=0.01):
        super(Decoder, self).__init__()

        # MLP layers to decode latent representation
        self.mlp_layers = nn.ModuleList()
        self.mlp_layers.append(nn.Linear(latent_dim, decoder_hidden_channels))
        for _ in range(num_layers - 1):
            self.mlp_layers.append(nn.Linear(decoder_hidden_channels, decoder_hidden_channels))

        # Output layer (expanding the latent space to the original dimension)
        self.final_layer = nn.Linear(decoder_hidden_channels, out_channels)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, Z):
        h = Z
        # MLP layers to decode the latent representation
        for layer in self.mlp_layers:
            h = F.leaky_relu(layer(h), negative_slope=0.2)
            h = self.dropout(h)

        X_recon = self.final_layer(h)  # Final layer to reconstruct the output
        return X_recon


class GraphAnomalyDetectionModel:
    def __init__(self, num_nodes=100, num_features=10, num_bins=4, latent_dim=8, encoder_hidden_channels=8, decoder_hidden_channels=32, num_layers=1, dropout_prob=0.01, lr=1e-3, epochs=1000, model_dir="models"):
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_bins = num_bins
        self.latent_dim = latent_dim
        self.encoder_hidden_channels = encoder_hidden_channels
        self.decoder_hidden_channels = decoder_hidden_channels
        self.lr = lr
        self.epochs = epochs
        self.model_dir = model_dir  # 模型文件夹路径
        self.in_channels = num_features
        self.out_channels = num_features
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        # 检查模型文件夹是否存在，不存在则创建
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.optimizer = None
        self.criterion_recon = nn.MSELoss()

        self.kmeans = None
        self.encoder_one_hot = None

        # 保存模型超参数
        self.model_params = {
            "num_nodes": self.num_nodes,
            "num_features": self.num_features,
            "num_bins": self.num_bins,
            "latent_dim": self.latent_dim,
            "encoder_hidden_channels": self.encoder_hidden_channels,
            "decoder_hidden_channels": self.decoder_hidden_channels,
            "lr": self.lr,
            "epochs": self.epochs,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
        }

        # 模型组件将在 fit 方法中初始化
        self.encoder = None
        self.decoder = None

        # 用于存储训练时每个节点和每个特征的重构误差，用于计算均值和方差
        self.node_recon_errors = {}  # 字典，键为节点位置索引，值为重构误差列表
        self.feature_recon_errors = {}  # 字典，键为节点位置索引，值为每个特征的重构误差列表

        # 数据增强模型
        self.xgb_model = None  # 使用 GradientBoostingRegressor

        # Store scaling parameters and split_values
        self.X_max = None
        self.X_min = None
        self.X_range = None
        self.split_values = None  # (num_bins + 1,)
        self.X_range = None

    # 辅助函数
    def filter_extreme_values(self, y_pred):
        """
        去除预测结果的 5%-95% 之外的极端值
        """
        lower_percentile = np.percentile(y_pred, 5)
        upper_percentile = np.percentile(y_pred, 95)
        return y_pred[(y_pred >= lower_percentile) & (y_pred <= upper_percentile)]

    def initialize_models(self, in_channels, out_channels):
        """根据数据维度初始化模型组件"""
        logger.debug("Initializing models...")
        self.encoder = GATEncoder(in_channels=in_channels, encoder_hidden_channels=self.encoder_hidden_channels, latent_dim=self.latent_dim, num_layers=self.num_layers, dropout_prob=self.dropout_prob).to(self.device)
        self.decoder = GATDecoder(latent_dim=self.latent_dim, decoder_hidden_channels=self.decoder_hidden_channels, out_channels=out_channels, num_layers=self.num_layers, dropout_prob=self.dropout_prob).to(self.device)
        # 设置优化器
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)
        logger.debug("Models initialized.")

    def feature_enhancement(self, data_list, fit=True):
        """对批量图数据进行特征增强，按节点特征进行 Min-Max 归一化"""
        # 将所有图的节点特征合并为一个数组
        X = np.array([data.x.numpy() for data in data_list])  # shape: (num_graphs, num_nodes, num_features)
        logger.debug(f"Feature Enhancement: Combined X shape: {X.shape}")

        num_graphs = X.shape[0]
        num_features = X.shape[2]

        if num_features != self.num_features:
            logger.warning(f"Number of features in data ({num_features}) does not match expected ({self.num_features}).")
            self.num_features = num_features

        if fit:
            # 计算每个节点和每个特征在所有图上的最小值和最大值
            self.X_min = X.min(axis=0)  # shape: (num_nodes, num_features)
            self.X_max = X.max(axis=0)  # shape: (num_nodes, num_features)
            # 处理除以零的情况
            self.X_range = self.X_max - self.X_min
            self.X_range[self.X_range == 0] = 1e-6
            # 应用 Min-Max 归一化
            X_norm = (X - self.X_min) / self.X_range  # shape: (num_graphs, num_nodes, num_features)

            # 特征增强：使用第1号和第2号节点的特征[0]来拟合第0号节点的特征[0]
            if self.num_nodes < 3:
                raise ValueError("num_nodes must be at least 3 for node1 and node2 features.")

            X_node1 = X_norm[:, 1, 0]  # 每个图的第1号节点的特征0 (shape: [num_graphs,])
            X_node2 = X_norm[:, 2, 0]  # 每个图的第2号节点的特征0 (shape: [num_graphs,])
            y = X_norm[:, 0, 0]  # 每个图的第0号节点的特征0 (shape: [num_graphs,])

            # X_node1 = X[:, 35, 0]  # 每个图的第1号节点的特征0 (shape: [num_graphs,])
            # X_node2 = X[:, 36, 0]  # 每个图的第2号节点的特征0 (shape: [num_graphs,])
            # y = X[:, 26, 0]  # 每个图的第0号节点的特征0 (shape: [num_graphs,])

            logger.debug(f"Feature Enhancement: X_node1 shape: {X_node1.shape}")
            logger.debug(f"Feature Enhancement: X_node2 shape: {X_node2.shape}")
            logger.debug(f"Feature Enhancement: y shape: {y.shape}")

            # 训练 GradientBoostingRegressor 模型（作为 XGBoost 的替代品）
            X_features = np.column_stack((X_node1, X_node2))  # shape: (num_graphs, 2)
            logger.debug(f"Feature Enhancement: X_features shape: {X_features.shape}")

            self.xgb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            self.xgb_model.fit(X_features, y)
            logger.debug("Feature Enhancement: GradientBoostingRegressor trained.")

            # Predict on training set to calculate errors
            y_pred = self.xgb_model.predict(X_features)  # shape: (num_graphs,)
            y_pred_filtered = self.filter_extreme_values(y_pred)
            logger.debug(f"Feature Enhancement: Filtered predictions shape: {y_pred_filtered.shape}")

            # Compute split_values as percentiles based on filtered predictions
            percentiles = np.percentile(y_pred_filtered, np.linspace(0, 100, self.num_bins + 1))
            self.split_values = percentiles  # shape: (num_bins + 1,)
            logger.debug(f"Feature Enhancement: Split values: {self.split_values}")

            # Classify y_pred into bins
            y_classified = np.digitize(y_pred, self.split_values[1:-1])  # 分类结果 (0到num_bins-1)
            logger.debug(f"Feature Enhancement: y_classified shape: {y_classified.shape}")
            logger.debug(f"Feature Enhancement: y_classified: {y_classified}")

            # 生成 One-Hot 编码
            class_features = np.eye(self.num_bins)[y_classified]  # shape: (num_graphs, num_bins)
            logger.debug(f"Feature Enhancement: class_features shape: {class_features.shape}")
        else:
            # 应用 Min-Max 归一化
            X_norm = (X - self.X_min) / self.X_range  # shape: (num_graphs, num_nodes, num_features)
            logger.debug(f"Feature Enhancement: X_norm (test) shape: {X_norm.shape}")

            # # 特征增强：使用第1号和第2号节点的特征0来预测第0号节点的特征0
            X_node1 = X_norm[:, 1, 0]  # 每个图的第1号节点的特征0 (shape: [num_graphs,])
            X_node2 = X_norm[:, 2, 0]  # 每个图的第2号节点的特征0 (shape: [num_graphs,])
            X_features = np.column_stack((X_node1, X_node2))  # shape: (num_graphs, 2)
            logger.debug(f"Feature Enhancement: X_features (test) shape: {X_features.shape}")

            # X_node1 = X[:, 35, 0]  # 每个图的第1号节点的特征0 (shape: [num_graphs,])
            # X_node2 = X[:, 36, 0]  # 每个图的第2号节点的特征0 (shape: [num_graphs,])
            # X_features = np.column_stack((X_node1, X_node2))  # shape: (num_graphs, 2)

            # 预测
            X_pred = self.xgb_model.predict(X_features)  # shape: (num_graphs,)
            y_true = X[:, 0, 0]  # shape: (num_graphs,)

            # 根据 split_values 进行分类
            y_classified = np.digitize(X_pred, self.split_values[1:-1])  # shape: (num_graphs,)
            logger.debug(f"Feature Enhancement: y_classified (test) shape: {y_classified.shape}")
            logger.debug(f"Feature Enhancement: y_classified (test): {y_classified}")

            # 生成 One-Hot 编码
            class_features = np.eye(self.num_bins)[y_classified]  # shape: (num_graphs, num_bins)
            logger.debug(f"Feature Enhancement: class_features (test) shape: {class_features.shape}")

        # 扩展 class_features 以匹配每个节点的特征
        class_features_expanded = np.repeat(class_features[:, np.newaxis, :], self.num_nodes, axis=1)  # shape: (num_graphs, num_nodes, num_bins)
        logger.debug(f"Feature Enhancement: class_features_expanded shape: {class_features_expanded.shape}")

        # 合并增强特征
        X_prime = np.concatenate((X_norm, class_features_expanded), axis=-1)  # shape: (num_graphs, num_nodes, num_features + num_bins)
        # X_prime = np.concatenate((X, class_features_expanded), axis=-1)  # shape: (num_graphs, num_nodes, num_features + num_bins)
        logger.debug(f"Feature Enhancement: X_prime shape after concatenation: {X_prime.shape}")

        # 将增强后的特征重新分配给各个图
        for i, data in enumerate(data_list):
            data.x_origin = data.x.clone()  # 保留原始特征，重命名为 x_origin
            data.x = torch.tensor(X_prime[i, :, :], dtype=torch.float)  # 显式切片，或者直接使用 X_prime[i]

        return data_list

    def fit(self, train_data):
        """训练模型"""
        # 进行特征增强
        enhanced_train_data = self.feature_enhancement(train_data, fit=True)
        # for id, data in enumerate(enhanced_train_data):
        #     enhanced_train_data[id].x = enhanced_train_data[id].x[:, 0].unsqueeze(-1)  # shape: (num_nodes, 1)

        # Determine in_channels and out_channels from data_list
        if len(enhanced_train_data) == 0:
            logger.error("No training data available after feature enhancement.")
            return

        self.in_channels = enhanced_train_data[0].x.size(1)
        self.out_channels = self.num_features

        self.model_params["in_channels"] = self.in_channels
        self.model_params["out_channels"] = self.out_channels

        # 初始化模型
        self.initialize_models(self.in_channels, self.out_channels)

        # 创建内部 DataLoader
        self.train_loader = DataLoader(enhanced_train_data, batch_size=1, shuffle=False)  # 每个 batch 是一个图
        logger.debug("Starting training...")

        # 设置模型为训练模式
        self.encoder.train()
        self.decoder.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_idx, batch_data in enumerate(self.train_loader):
                # 获取增强后的特征和边
                X_prime = batch_data.x.to(self.device)  # shape: (num_nodes, num_features + num_bins)
                X_original = batch_data.x_origin.to(self.device)  # shape: (num_nodes, num_features)
                edge_index = batch_data.edge_index.to(self.device)

                self.optimizer.zero_grad()

                # Encoder
                Z = self.encoder(X_prime, edge_index)  # Latent space representation

                # Decoder
                X_recon = self.decoder(Z, edge_index)

                # logger.debug(f"fit X_recon: {X_recon[:, : self.out_channels]}")
                # logger.debug(f"fit X_prime: {X_prime}")

                # 计算原始特征的重构误差
                loss_recon = self.criterion_recon(X_recon, X_prime[:, : self.out_channels])

                # 总损失
                loss_total = loss_recon

                # 反向传播
                loss_total.backward()
                self.optimizer.step()

                total_loss += loss_total.item()

            avg_loss = total_loss / len(self.train_loader)
            if (epoch + 1) % 1 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.8f}")
                # logging.info(f"X_recon: {X_recon[:5]}")
                # logging.info(f"X_prime: {X_prime[:5]}")

        # 训练完成后进行预测并记录重构误差
        self.evaluate(enhanced_train_data)

    def evaluate(self, data_list):
        """在训练完成后对训练数据进行预测，并记录每个节点和每个特征的重构误差"""
        self.encoder.eval()
        self.decoder.eval()
        self.eval_loader = DataLoader(data_list, batch_size=1, shuffle=False)

        with torch.no_grad():
            for i, data in enumerate(self.eval_loader):
                # 获取增强后的特征和边
                X_prime = data.x.to(self.device)  # shape: (num_nodes, num_features + num_bins)
                X_original = data.x_origin.to(self.device)  # shape: (num_nodes, num_features)
                edge_index = data.edge_index.to(self.device)

                # 编码器
                Z = self.encoder(X_prime, edge_index)
                # 解码器
                X_recon = self.decoder(Z, edge_index)

                X_recon_inverse = X_recon.cpu().numpy() * self.X_range + self.X_min
                # X_recon_inverse = X_recon.cpu().numpy()

                X_original = X_original.cpu().detach().numpy()

                # X_original = X_prime[:, : self.out_channels].cpu().detach().numpy()

                # 计算原始特征的重构误差
                recon_errors_node = np.sum((X_recon_inverse - X_original) ** 2, axis=1)  # shape: (num_nodes,)
                recon_errors_feature = (X_recon_inverse - X_original) ** 2  # shape: (num_nodes, num_features)
                # 记录每个节点的重构误差
                for node_pos in range(data.num_nodes):
                    error_node = recon_errors_node[node_pos]
                    if node_pos not in self.node_recon_errors:
                        self.node_recon_errors[node_pos] = []
                    self.node_recon_errors[node_pos].append(error_node)

                    # 记录每个特征的重构误差
                    if node_pos not in self.feature_recon_errors:
                        self.feature_recon_errors[node_pos] = [[] for _ in range(self.out_channels)]
                    for feature_pos in range(self.out_channels):
                        error_feature = recon_errors_feature[node_pos, feature_pos]
                        self.feature_recon_errors[node_pos][feature_pos].append(error_feature)

        logger.debug("Evaluation completed.")

    def predict(self, test_data):
        """使用训练好的模型对新数据进行预测，返回所有图所有节点所有特征的预测重构结果"""
        self.load_models()
        # 进行特征增强
        enhanced_test_data = self.feature_enhancement(test_data, fit=False)
        # 创建内部 DataLoader
        test_loader = DataLoader(enhanced_test_data, batch_size=1, shuffle=False)  # 每个 batch 是一个图
        logger.debug("Starting prediction...")

        self.encoder.eval()
        self.decoder.eval()

        all_reconstructions = []

        for batch_idx, batch_data in enumerate(test_loader):
            # 获取增强后的特征和边
            X_prime = batch_data.x.to(self.device)  # shape: (num_nodes, num_features + num_bins)
            edge_index = batch_data.edge_index.to(self.device)

            with torch.no_grad():
                # 编码器
                Z = self.encoder(X_prime, edge_index)

                # 解码器
                X_recon = self.decoder(Z, edge_index)

                # 保存重构结果
                X_recon_inverse = X_recon.cpu().numpy() * self.X_range + self.X_min
                # X_recon_inverse = X_recon.cpu().numpy()
                all_reconstructions.append(X_recon_inverse)

                # logger.debug(f"predict X_recon: {X_recon}")
                # logger.debug(f"predict X_recon_inverse: {X_recon_inverse}")

        # 将所有重构结果组合成一个数组，形状为 (图数, 节点数, 特征数)
        all_reconstructions = np.stack(all_reconstructions, axis=0)
        return all_reconstructions

    def save_models(self):
        """保存模型和组件"""
        # 保存模型和组件
        logger.debug("Saving models and components...")
        torch.save(self.encoder.state_dict(), os.path.join(self.model_dir, "encoder.pth"))
        torch.save(self.decoder.state_dict(), os.path.join(self.model_dir, "decoder.pth"))
        joblib.dump(
            {
                "X_max": self.X_max,
                "X_min": self.X_min,
                "X_range": self.X_range,
                "split_values": self.split_values,
                "xgb_model": self.xgb_model,
                "node_recon_errors": self.node_recon_errors,
                "feature_recon_errors": self.feature_recon_errors,
            },
            os.path.join(self.model_dir, "model_components.pkl"),
        )
        with open(os.path.join(self.model_dir, "model_params.json"), "w") as f:
            json.dump(self.model_params, f)
        logger.debug("Models and components saved successfully.")

    def load_models(self):
        """加载模型和组件"""
        if os.path.exists(self.model_dir):
            logger.debug("Loading models and components...")
            # 加载模型超参数
            with open(os.path.join(self.model_dir, "model_params.json"), "r") as f:
                self.model_params = json.load(f)

            # 加载其他组件
            components = joblib.load(os.path.join(self.model_dir, "model_components.pkl"))
            self.X_max = components["X_max"]
            self.X_min = components["X_min"]
            self.X_range = components["X_range"]
            self.split_values = components["split_values"]
            self.xgb_model = components["xgb_model"]
            self.node_recon_errors = components["node_recon_errors"]
            self.feature_recon_errors = components["feature_recon_errors"]
            logger.debug(f"Loaded split_values: {self.split_values}")
            self.in_channels = self.model_params["in_channels"]
            self.out_channels = self.model_params["out_channels"]

            # 初始化模型
            self.initialize_models(self.model_params["in_channels"], self.model_params["out_channels"])

            # 加载模型权重
            self.encoder.load_state_dict(torch.load(os.path.join(self.model_dir, "encoder.pth")))
            self.decoder.load_state_dict(torch.load(os.path.join(self.model_dir, "decoder.pth")))

            logger.debug("Model and components loaded successfully.")
        else:
            logger.error(f"Model directory {self.model_dir} not found.")
            return None

    def compute_anomaly_scores(self, test_data, reconstructions):
        """根据预测的重构结果计算z-score，并返回所需的四个列表"""
        # 初始化字典存储重构误差
        test_node_recon_errors = {}
        test_feature_recon_errors = {}

        # 初始化列表存储所有图的重构误差
        test_node_z_scores = []  # shape: (num_graphs, num_nodes)
        test_feature_z_scores = []  # shape: (num_graphs, num_nodes, num_features)

        # 计算每个图的重构误差
        for graph_idx, (batch_data, X_recon) in enumerate(zip(test_data, reconstructions)):
            X_original = batch_data.x_origin.cpu().numpy()  # shape: (num_nodes, num_features)
            # X_original = batch_data.x[:, : self.out_channels].cpu().numpy()  # shape: (num_nodes, num_features)

            recon_errors_node = np.sum((X_recon - X_original) ** 2, axis=1)  # shape: (num_nodes,)
            recon_errors_feature = (X_recon - X_original) ** 2  # shape: (num_nodes, num_features)
            # 初始化当前图的 z-score 列表
            current_graph_node_z_scores = []
            current_graph_feature_z_scores = []
            # 计算每个节点的z-score
            for node_pos in range(batch_data.num_nodes):
                error_node = recon_errors_node[node_pos]
                mean_node = np.mean(self.node_recon_errors[node_pos])
                std_node = np.std(self.node_recon_errors[node_pos])
                if std_node == 0:
                    std_node = 1e-8
                node_z_score = (error_node - mean_node) / std_node
                # node_z_score = error_node
                current_graph_node_z_scores.append(node_z_score)

                current_node_feature_z_scores = []
                for feature_pos in range(self.out_channels):
                    error_feature = recon_errors_feature[node_pos, feature_pos]
                    mean_feature = np.mean(self.feature_recon_errors[node_pos][feature_pos])
                    std_feature = np.std(self.feature_recon_errors[node_pos][feature_pos])
                    if std_feature == 0:
                        std_feature = 1e-8
                    feature_z_score = (error_feature - mean_feature) / std_feature
                    current_node_feature_z_scores.append(feature_z_score)

                current_graph_feature_z_scores.append(current_node_feature_z_scores)
            test_node_z_scores.append(current_graph_node_z_scores)
            test_feature_z_scores.append(current_graph_feature_z_scores)
        return test_node_z_scores, test_feature_z_scores


def create_synthetic_data(num_graphs=100, num_nodes=100, num_features=10, num_clusters=3):
    """生成合成图数据"""
    data_list = []
    for _ in range(num_graphs):
        X = np.random.randn(num_nodes, num_features)
        X = 5 + 3 * X
        A = np.random.randint(0, 2, size=(num_nodes, num_nodes))
        A = np.triu(A, 1)
        A = A + A.T  # 对称邻接矩阵
        edge_index = dense_to_sparse(torch.tensor(A, dtype=torch.float))[0]
        data = Data(x=torch.tensor(X, dtype=torch.float), edge_index=edge_index)
        data_list.append(data)
    return data_list


# # 主要训练和预测部分
# if __name__ == "__main__":
#     # 创建合成数据
#     num_graphs = 100  # 样本数量
#     num_nodes = 5  # 每个图的节点数量
#     num_features = 10  # 每个节点的特征维度
#     num_clusters = 3  # 聚类数量

#     data_list = create_synthetic_data(num_graphs=num_graphs, num_nodes=num_nodes, num_features=num_features, num_clusters=num_clusters)
#     train_data = data_list[:80]  # 80个图用于训练
#     test_data = data_list[80:]  # 20个图用于测试

#     # 初始化模型
#     model = GraphAnomalyDetectionModel(num_nodes=num_nodes, num_features=num_features, num_bins=4, latent_dim=8, hidden_channels=16, epochs=2, model_dir="models")  # 减少epoch数量以加快示例

#     # 训练模型
#     model.node_recon_errors = {}  # 初始化重构误差字典
#     model.feature_recon_errors = {}  # 初始化特征重构误差字典
#     model.fit(train_data)

#     # 保存模型
#     model.save_models()

#     # 预测模型
#     reconstructions = model.predict(test_data)
#     test_node_z_scores, test_feature_z_scores = model.compute_anomaly_scores(test_data, reconstructions)

#     logging.debug(f"test_node_z_scores: {test_node_z_scores}")
#     logging.debug(f"test_feature_z_scores: {test_feature_z_scores}")
