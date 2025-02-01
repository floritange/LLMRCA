import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import GradientBoostingRegressor  # Use a substitute for XGBoost
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dense_to_sparse
import logging
import joblib  # For saving and loading model components
import os
import json
import torch.nn.functional as F

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(name)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Output to console
)

logger = logging.getLogger(__name__)

# """ Regular GAT without Res """
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


""" GCN """


class GATEncoder(nn.Module):
    def __init__(self, in_channels, encoder_hidden_channels, latent_dim, num_layers=3, dropout_prob=0.01):
        super(GATEncoder, self).__init__()
        # Linear transformation of input features x to adjust dimensions
        self.linear_input = nn.Linear(in_channels, encoder_hidden_channels)
        self.gcn_layers = nn.ModuleList()
        self.linear_skip_layers = nn.ModuleList()
        # Subsequent GCN layers: each layer takes encoder_hidden_channels as input and produces encoder_hidden_channels as output
        for _ in range(num_layers):
            self.gcn_layers.append(GCNConv(encoder_hidden_channels, encoder_hidden_channels))
            self.linear_skip_layers.append(nn.Linear(encoder_hidden_channels, latent_dim))
        # Latent layer: the final GCN layer produces latent_dim
        self.latent_layer = GCNConv(encoder_hidden_channels, latent_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        # First apply linear transformation to input features x, adjusting to encoder_hidden_channels
        h = self.linear_input(x)
        skip_connections = []

        # Apply GCN to each layer and record skip connections
        for layer in self.gcn_layers:
            skip_connections.append(h)  # Record the output of the current layer (skip connection)
            h = self.leaky_relu(layer(h, edge_index))  # Apply GCNConv and LeakyReLU
            h = self.dropout(h)  # Apply Dropout
        # The final latent representation
        latent = self.latent_layer(h, edge_index)
        # Use linear layers to adjust the dimensions of each skip connection to match latent
        for i in range(len(skip_connections)):
            skip_connections[i] = self.linear_skip_layers[i](skip_connections[i])
        # Combine all skip connections with latent
        for skip in skip_connections:
            latent = latent + skip
        return latent


class GATDecoder(nn.Module):
    def __init__(self, latent_dim, decoder_hidden_channels, out_channels, num_layers=3, dropout_prob=0.01):
        super(GATDecoder, self).__init__()

        self.linear_input = nn.Linear(latent_dim, decoder_hidden_channels)
        self.gcn_layers = nn.ModuleList()
        self.linear_skip_layers = nn.ModuleList()
        # GCN layers: each layer takes decoder_hidden_channels as input and produces decoder_hidden_channels as output
        for _ in range(num_layers):
            self.gcn_layers.append(GCNConv(decoder_hidden_channels, decoder_hidden_channels))
            self.linear_skip_layers.append(nn.Linear(decoder_hidden_channels, out_channels))  # Corresponding linear layers

        # Reconstruction output layer, finally output out_channels dimension
        self.final_layer = nn.Linear(decoder_hidden_channels, out_channels)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, z, edge_index):
        h = self.linear_input(z)
        skip_connections = []

        # Apply GCN to each layer and record skip connections
        for layer in self.gcn_layers:
            skip_connections.append(h)
            h = F.leaky_relu(layer(h, edge_index), negative_slope=0.2)
            h = self.dropout(h)  # Apply Dropout
        # The final reconstruction output
        x_recon = self.final_layer(h)
        # Use linear layers to adjust the dimensions of each skip connection to match x_recon
        for i in range(len(skip_connections)):
            skip_connections[i] = self.linear_skip_layers[i](skip_connections[i])
        # Combine all skip connections with x_recon
        for skip in skip_connections:
            x_recon = x_recon + skip
        return x_recon


""" Original GAT """
# class GATEncoder(nn.Module):
#     def __init__(self, in_channels, encoder_hidden_channels, latent_dim, num_layers=3, heads=1, dropout_prob=0.01):
#         super(GATEncoder, self).__init__()
#         self.linear_input = nn.Linear(in_channels, encoder_hidden_channels * heads)
#         self.gat_layers = nn.ModuleList()
#         self.linear_skip_layers = nn.ModuleList()
#         for _ in range(num_layers):
#             self.gat_layers.append(GATConv(encoder_hidden_channels * heads, encoder_hidden_channels, heads=heads, concat=True))
#             self.linear_skip_layers.append(nn.Linear(encoder_hidden_channels * heads, latent_dim))
#         self.latent_layer = GATConv(encoder_hidden_channels * heads, latent_dim, heads=1, concat=False)
#         self.dropout = nn.Dropout(p=dropout_prob)
#         self.leaky_relu = nn.LeakyReLU(0.2)

#     def forward(self, x, edge_index):
#         h = self.linear_input(x)
#         skip_connections = []
#         for layer in self.gat_layers:
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
#     def __init__(self, latent_dim, decoder_hidden_channels, out_channels, num_layers=3, heads=1, dropout_prob=0.01):
#         super(GATDecoder, self).__init__()
#         self.linear_input = nn.Linear(latent_dim, decoder_hidden_channels * heads)
#         self.gat_layers = nn.ModuleList()
#         self.linear_skip_layers = nn.ModuleList()
#         for _ in range(num_layers):
#             self.gat_layers.append(GATConv(decoder_hidden_channels * heads, decoder_hidden_channels, heads=heads, concat=True))
#             self.linear_skip_layers.append(nn.Linear(decoder_hidden_channels * heads, out_channels))
#         self.final_layer = nn.Linear(decoder_hidden_channels * heads, out_channels)
#         self.dropout = nn.Dropout(p=dropout_prob)

#     def forward(self, z, edge_index):
#         h = self.linear_input(z)
#         skip_connections = []
#         for layer in self.gat_layers:
#             skip_connections.append(h)
#             h = F.leaky_relu(layer(h, edge_index), negative_slope=0.2)
#             h = self.dropout(h)
#         x_recon = self.final_layer(h)
#         for i in range(len(skip_connections)):
#             skip_connections[i] = self.linear_skip_layers[i](skip_connections[i])
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
    def __init__(
        self, num_nodes=100, num_features=10, num_bins=4, latent_dim=8, encoder_hidden_channels=8, decoder_hidden_channels=32, num_layers=1, dropout_prob=0.01, lr=1e-3, epochs=1000, model_dir="models"
    ):
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_bins = num_bins
        self.latent_dim = latent_dim
        self.encoder_hidden_channels = encoder_hidden_channels
        self.decoder_hidden_channels = decoder_hidden_channels
        self.lr = lr
        self.epochs = epochs
        self.model_dir = model_dir  # Model folder path
        self.in_channels = num_features
        self.out_channels = num_features
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        # Check if model folder exists, if not, create it
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.optimizer = None
        self.criterion_recon = nn.MSELoss()

        self.kmeans = None
        self.encoder_one_hot = None

        # Save model hyperparameters
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

        # Model components will be initialized in the fit method
        self.encoder = None
        self.decoder = None

        # Used to store reconstruction errors for each node and feature during training to compute mean and variance
        self.node_recon_errors = {}  # Dictionary with node positions as keys and reconstruction errors as lists
        self.feature_recon_errors = {}  # Dictionary with node positions as keys and feature reconstruction errors as lists

        # Data augmentation model
        self.xgb_model = None  # Using GradientBoostingRegressor

        # Store scaling parameters and split_values
        self.X_max = None
        self.X_min = None
        self.X_range = None
        self.split_values = None  # (num_bins + 1,)
        self.X_range = None

    # Helper functions
    def filter_extreme_values(self, y_pred):
        """
        Remove extreme values outside of the 5%-95% range from the predicted results
        """
        lower_percentile = np.percentile(y_pred, 5)
        upper_percentile = np.percentile(y_pred, 95)
        return y_pred[(y_pred >= lower_percentile) & (y_pred <= upper_percentile)]

    def initialize_models(self, in_channels, out_channels):
        """Initialize model components based on data dimensions"""
        logger.debug("Initializing models...")
        self.encoder = GATEncoder(
            in_channels=in_channels, encoder_hidden_channels=self.encoder_hidden_channels, latent_dim=self.latent_dim, num_layers=self.num_layers, dropout_prob=self.dropout_prob
        ).to(self.device)
        self.decoder = GATDecoder(
            latent_dim=self.latent_dim, decoder_hidden_channels=self.decoder_hidden_channels, out_channels=out_channels, num_layers=self.num_layers, dropout_prob=self.dropout_prob
        ).to(self.device)
        # Set the optimizer
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)
        logger.debug("Models initialized.")

    def feature_enhancement(self, data_list, fit=True):
        """Perform feature enhancement on a batch of graph data using Min-Max normalization on node features"""
        # Combine all node features from all graphs into one array
        X = np.array([data.x.numpy() for data in data_list])  # shape: (num_graphs, num_nodes, num_features)
        logger.debug(f"Feature Enhancement: Combined X shape: {X.shape}")

        num_graphs = X.shape[0]
        num_features = X.shape[2]

        if num_features != self.num_features:
            logger.warning(f"Number of features in data ({num_features}) does not match expected ({self.num_features}).")
            self.num_features = num_features

        if fit:
            # Calculate the minimum and maximum values for each node and feature across all graphs
            self.X_min = X.min(axis=0)  # shape: (num_nodes, num_features)
            self.X_max = X.max(axis=0)  # shape: (num_nodes, num_features)
            # Handle division by zero
            self.X_range = self.X_max - self.X_min
            self.X_range[self.X_range == 0] = 1e-6
            # Apply Min-Max normalization
            X_norm = (X - self.X_min) / self.X_range  # shape: (num_graphs, num_nodes, num_features)

            # Feature enhancement: use features of node 1 and node 2 to fit features of node 0
            if self.num_nodes < 3:
                raise ValueError("num_nodes must be at least 3 for node1 and node2 features.")

            X_node1 = X_norm[:, 1, 0]  # Feature 0 of node 1 in each graph (shape: [num_graphs,])
            X_node2 = X_norm[:, 2, 0]  # Feature 0 of node 2 in each graph (shape: [num_graphs,])
            y = X_norm[:, 0, 0]  # Feature 0 of node 0 in each graph (shape: [num_graphs,])

            logger.debug(f"Feature Enhancement: X_node1 shape: {X_node1.shape}")
            logger.debug(f"Feature Enhancement: X_node2 shape: {X_node2.shape}")
            logger.debug(f"Feature Enhancement: y shape: {y.shape}")

            # Train GradientBoostingRegressor model (as a substitute for XGBoost)
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
            y_classified = np.digitize(y_pred, self.split_values[1:-1])  # Classification result (0 to num_bins-1)
            logger.debug(f"Feature Enhancement: y_classified shape: {y_classified.shape}")
            logger.debug(f"Feature Enhancement: y_classified: {y_classified}")

            # Generate One-Hot encoding
            class_features = np.eye(self.num_bins)[y_classified]  # shape: (num_graphs, num_bins)
            logger.debug(f"Feature Enhancement: class_features shape: {class_features.shape}")
        else:
            # Apply Min-Max normalization
            X_norm = (X - self.X_min) / self.X_range  # shape: (num_graphs, num_nodes, num_features)
            logger.debug(f"Feature Enhancement: X_norm (test) shape: {X_norm.shape}")

            # Feature enhancement: use features of node 1 and node 2 to predict features of node 0
            X_node1 = X_norm[:, 1, 0]  # Feature 0 of node 1 in each graph (shape: [num_graphs,])
            X_node2 = X_norm[:, 2, 0]  # Feature 0 of node 2 in each graph (shape: [num_graphs,])
            X_features = np.column_stack((X_node1, X_node2))  # shape: (num_graphs, 2)
            logger.debug(f"Feature Enhancement: X_features (test) shape: {X_features.shape}")

            # Predict
            X_pred = self.xgb_model.predict(X_features)  # shape: (num_graphs,)
            y_true = X[:, 0, 0]  # shape: (num_graphs,)

            # Classify based on split_values
            y_classified = np.digitize(X_pred, self.split_values[1:-1])  # shape: (num_graphs,)
            logger.debug(f"Feature Enhancement: y_classified (test) shape: {y_classified.shape}")
            logger.debug(f"Feature Enhancement: y_classified (test): {y_classified}")

            # Generate One-Hot encoding
            class_features = np.eye(self.num_bins)[y_classified]  # shape: (num_graphs, num_bins)
            logger.debug(f"Feature Enhancement: class_features (test) shape: {class_features.shape}")

        # Expand class_features to match each node's features
        class_features_expanded = np.repeat(class_features[:, np.newaxis, :], self.num_nodes, axis=1)  # shape: (num_graphs, num_nodes, num_bins)
        logger.debug(f"Feature Enhancement: class_features_expanded shape: {class_features_expanded.shape}")

        # Combine enhanced features
        X_prime = np.concatenate((X_norm, class_features_expanded), axis=-1)  # shape: (num_graphs, num_nodes, num_features + num_bins)
        # X_prime = np.concatenate((X, class_features_expanded), axis=-1)  # shape: (num_graphs, num_nodes, num_features + num_bins)
        logger.debug(f"Feature Enhancement: X_prime shape after concatenation: {X_prime.shape}")

        # Reassign enhanced features to each graph
        for i, data in enumerate(data_list):
            data.x_origin = data.x.clone()  # Keep original features, renamed as x_origin
            data.x = torch.tensor(X_prime[i, :, :], dtype=torch.float)  # Explicitly slice, or directly use X_prime[i]

        return data_list

    def fit(self, train_data):
        """Train the model"""
        # Perform feature enhancement
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

        # Initialize models
        self.initialize_models(self.in_channels, self.out_channels)

        # Create internal DataLoader
        self.train_loader = DataLoader(enhanced_train_data, batch_size=1, shuffle=False)  # Each batch is one graph
        logger.debug("Starting training...")

        # Set models to training mode
        self.encoder.train()
        self.decoder.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_idx, batch_data in enumerate(self.train_loader):
                # Get enhanced features and edges
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

                # Calculate reconstruction error for original features
                loss_recon = self.criterion_recon(X_recon, X_prime[:, : self.out_channels])

                # Total loss
                loss_total = loss_recon

                # Backpropagation
                loss_total.backward()
                self.optimizer.step()

                total_loss += loss_total.item()

            avg_loss = total_loss / len(self.train_loader)
            if (epoch + 1) % 1 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.8f}")
                # logging.info(f"X_recon: {X_recon[:5]}")
                # logging.info(f"X_prime: {X_prime[:5]}")

        # After training, perform prediction and record reconstruction errors
        self.evaluate(enhanced_train_data)

    def evaluate(self, data_list):
        """After training, predict on training data and record reconstruction errors for each node and feature"""
        self.encoder.eval()
        self.decoder.eval()
        self.eval_loader = DataLoader(data_list, batch_size=1, shuffle=False)

        with torch.no_grad():
            for i, data in enumerate(self.eval_loader):
                # Get enhanced features and edges
                X_prime = data.x.to(self.device)  # shape: (num_nodes, num_features + num_bins)
                X_original = data.x_origin.to(self.device)  # shape: (num_nodes, num_features)
                edge_index = data.edge_index.to(self.device)

                # Encoder
                Z = self.encoder(X_prime, edge_index)
                # Decoder
                X_recon = self.decoder(Z, edge_index)

                X_recon_inverse = X_recon.cpu().numpy() * self.X_range + self.X_min
                # X_recon_inverse = X_recon.cpu().numpy()

                X_original = X_original.cpu().detach().numpy()

                # X_original = X_prime[:, : self.out_channels].cpu().detach().numpy()

                # Calculate reconstruction error for original features
                recon_errors_node = np.sum((X_recon_inverse - X_original) ** 2, axis=1)  # shape: (num_nodes,)
                recon_errors_feature = (X_recon_inverse - X_original) ** 2  # shape: (num_nodes, num_features)
                # Record reconstruction errors for each node
                for node_pos in range(data.num_nodes):
                    error_node = recon_errors_node[node_pos]
                    if node_pos not in self.node_recon_errors:
                        self.node_recon_errors[node_pos] = []
                    self.node_recon_errors[node_pos].append(error_node)

                    # Record reconstruction errors for each feature
                    if node_pos not in self.feature_recon_errors:
                        self.feature_recon_errors[node_pos] = [[] for _ in range(self.out_channels)]
                    for feature_pos in range(self.out_channels):
                        error_feature = recon_errors_feature[node_pos, feature_pos]
                        self.feature_recon_errors[node_pos][feature_pos].append(error_feature)

        logger.debug("Evaluation completed.")

    def predict(self, test_data):
        """Use the trained model to predict new data and return the predicted reconstruction results for all graphs, nodes, and features"""
        self.load_models()
        # Perform feature enhancement
        enhanced_test_data = self.feature_enhancement(test_data, fit=False)
        # Create internal DataLoader
        test_loader = DataLoader(enhanced_test_data, batch_size=1, shuffle=False)  # Each batch is one graph
        logger.debug("Starting prediction...")

        self.encoder.eval()
        self.decoder.eval()

        all_reconstructions = []

        for batch_idx, batch_data in enumerate(test_loader):
            # Get enhanced features and edges
            X_prime = batch_data.x.to(self.device)  # shape: (num_nodes, num_features + num_bins)
            edge_index = batch_data.edge_index.to(self.device)

            with torch.no_grad():
                # Encoder
                Z = self.encoder(X_prime, edge_index)

                # Decoder
                X_recon = self.decoder(Z, edge_index)

                # Save reconstruction results
                X_recon_inverse = X_recon.cpu().numpy() * self.X_range + self.X_min
                # X_recon_inverse = X_recon.cpu().numpy()
                all_reconstructions.append(X_recon_inverse)

                # logger.debug(f"predict X_recon: {X_recon}")
                # logger.debug(f"predict X_recon_inverse: {X_recon_inverse}")

        # Combine all reconstruction results into one array, shape: (num_graphs, num_nodes, num_features)
        all_reconstructions = np.stack(all_reconstructions, axis=0)
        return all_reconstructions

    def save_models(self):
        """Save the models and components"""
        # Save models and components
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
        """Load the models and components"""
        if os.path.exists(self.model_dir):
            logger.debug("Loading models and components...")
            # Load model hyperparameters
            with open(os.path.join(self.model_dir, "model_params.json"), "r") as f:
                self.model_params = json.load(f)

            # Load other components
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

            # Initialize models
            self.initialize_models(self.model_params["in_channels"], self.model_params["out_channels"])

            # Load model weights
            self.encoder.load_state_dict(torch.load(os.path.join(self.model_dir, "encoder.pth")))
            self.decoder.load_state_dict(torch.load(os.path.join(self.model_dir, "decoder.pth")))

            logger.debug("Model and components loaded successfully.")
        else:
            logger.error(f"Model directory {self.model_dir} not found.")
            return None

    def compute_anomaly_scores(self, test_data, reconstructions):
        """Compute z-scores based on predicted reconstruction results and return the required four lists"""
        # Initialize dictionaries to store reconstruction errors
        test_node_recon_errors = {}
        test_feature_recon_errors = {}

        # Initialize lists to store reconstruction errors for all graphs
        test_node_z_scores = []  # shape: (num_graphs, num_nodes)
        test_feature_z_scores = []  # shape: (num_graphs, num_nodes, num_features)

        # Compute reconstruction errors for each graph
        for graph_idx, (batch_data, X_recon) in enumerate(zip(test_data, reconstructions)):
            X_original = batch_data.x_origin.cpu().numpy()  # shape: (num_nodes, num_features)
            # X_original = batch_data.x[:, : self.out_channels].cpu().numpy()  # shape: (num_nodes, num_features)

            recon_errors_node = np.sum((X_recon - X_original) ** 2, axis=1)  # shape: (num_nodes,)
            recon_errors_feature = (X_recon - X_original) ** 2  # shape: (num_nodes, num_features)
            # Initialize z-score list for the current graph
            current_graph_node_z_scores = []
            current_graph_feature_z_scores = []
            # Compute z-scores for each node
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
    """Generate synthetic graph data"""
    data_list = []
    for _ in range(num_graphs):
        X = np.random.randn(num_nodes, num_features)
        X = 5 + 3 * X
        A = np.random.randint(0, 2, size=(num_nodes, num_nodes))
        A = np.triu(A, 1)
        A = A + A.T  # Symmetric adjacency matrix
        edge_index = dense_to_sparse(torch.tensor(A, dtype=torch.float))[0]
        data = Data(x=torch.tensor(X, dtype=torch.float), edge_index=edge_index)
        data_list.append(data)
    return data_list


# # Main training and prediction part
# if __name__ == "__main__":
#     # Create synthetic data
#     num_graphs = 100  # Number of samples
#     num_nodes = 5  # Number of nodes per graph
#     num_features = 10  # Number of features per node
#     num_clusters = 3  # Number of clusters

#     data_list = create_synthetic_data(num_graphs=num_graphs, num_nodes=num_nodes, num_features=num_features, num_clusters=num_clusters)
#     train_data = data_list[:80]  # 80 graphs for training
#     test_data = data_list[80:]  # 20 graphs for testing

#     # Initialize model
#     model = GraphAnomalyDetectionModel(num_nodes=num_nodes, num_features=num_features, num_bins=4, latent_dim=8, hidden_channels=16, epochs=2, model_dir="models")  # Reduce epoch count for faster example

#     # Train model
#     model.node_recon_errors = {}  # Initialize reconstruction error dictionary
#     model.feature_recon_errors = {}  # Initialize feature reconstruction error dictionary
#     model.fit(train_data)

#     # Save model
#     model.save_models()

#     # Predict with model
#     reconstructions = model.predict(test_data)
#     test_node_z_scores, test_feature_z_scores = model.compute_anomaly_scores(test_data, reconstructions)

#     logging.debug(f"test_node_z_scores: {test_node_z_scores}")
#     logging.debug(f"test_feature_z_scores: {test_feature_z_scores}")
