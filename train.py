import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import os
from scipy import stats
import pickle
import json
import time
import shap

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ======================= CONFIGURATION =======================
FEATURES_CSV = "feature_extracted_augment_dataset.csv"
GROUND_TRUTH_FILE = "augmented_ground_truth_959.xlsx"

OUTPUT_DIR = "EAAI_journal_results_complete_FINAL"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for subdir in ['models', 'tables', 'predictions', 'shap']:
    os.makedirs(f"{OUTPUT_DIR}/{subdir}", exist_ok=True)

SEEDS = [42, 123, 7, 2024, 88, 999, 101, 55, 303, 777, 1337, 2026, 11, 888, 555]  # Expanded to 15 seeds
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTLIER_THRESHOLD = 3.0
BATCH_SIZE = 32
EPOCHS = 150
PATIENCE = 25

print(f"Device: {DEVICE}")
print(f"Output Directory: {OUTPUT_DIR}")

# ==================== ALL YOUR ORIGINAL MODEL ARCHITECTURES ====================
# EXACT SAME AS YOUR ORIGINAL CODE - NO CHANGES

class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return output + self.bias

class DynamicGraphFeatureConvolution(nn.Module):
    def __init__(self, num_base_models, input_dim):
        super().__init__()
        self.num_nodes = input_dim 
        self.node_embedding = nn.Linear(1, 16) 
        self.graph_conv1 = GraphConvolutionLayer(16, 32)
        self.graph_conv2 = GraphConvolutionLayer(32, 64)
        self.meta_fusion = nn.Sequential(
            nn.Linear(num_base_models + (input_dim * 64), 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))
    def forward(self, meta_preds, raw_features):
        batch_size = raw_features.size(0)
        x_graph = raw_features.unsqueeze(-1) 
        x_emb = self.node_embedding(x_graph)
        Q = x_emb.mean(dim=2, keepdim=True) 
        K = x_emb.mean(dim=2, keepdim=True).permute(0, 2, 1)
        adj = F.softmax(torch.matmul(Q, K) / np.sqrt(16), dim=-1)
        x_g = F.relu(self.graph_conv1(x_emb, adj))
        x_g = F.relu(self.graph_conv2(x_g, adj))
        x_g_flat = x_g.view(batch_size, -1)
        combined = torch.cat([meta_preds, x_g_flat], dim=1)
        return self.meta_fusion(combined).squeeze()

class ResidualErrorCorrectingNetwork(nn.Module):
    def __init__(self, num_models, input_dim):
        super().__init__()
        self.context_encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32))
        self.error_predictor = nn.Sequential(
            nn.Linear(num_models + 32, 128), nn.LayerNorm(128), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
        self.selector = nn.Sequential(nn.Linear(input_dim + num_models, num_models), nn.Softmax(dim=1))
    def forward(self, meta_preds, raw_features):
        context = self.context_encoder(raw_features)
        selector_input = torch.cat([raw_features, meta_preds], dim=1)
        weights = self.selector(selector_input)
        anchor_prediction = (meta_preds * weights).sum(dim=1, keepdim=True)
        combiner = torch.cat([meta_preds, context], dim=1)
        predicted_error = self.error_predictor(combiner)
        final_pred = anchor_prediction + predicted_error
        return final_pred.squeeze()

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5):
        super().__init__()
        self.in_features, self.out_features, self.grid_size = in_features, out_features, grid_size
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.poly_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5))
        nn.init.normal_(self.poly_weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.bias)
    def forward(self, x):
        base_out = F.linear(x, self.base_weight)
        x_expanded = torch.stack([torch.sin(i * x) for i in range(1, self.grid_size + 1)], dim=-1)
        poly_out = torch.einsum('big,oig->bo', x_expanded, self.poly_weight)
        return F.silu(base_out + poly_out + self.bias)

class KANFusionNetwork(nn.Module):
    def __init__(self, num_models, input_dim):
        super().__init__()
        combined_dim = num_models + input_dim
        self.kan1 = KANLayer(combined_dim, 64)
        self.kan2 = KANLayer(64, 32)
        self.kan3 = KANLayer(32, 1)
        self.batch_norm = nn.BatchNorm1d(combined_dim)
    def forward(self, meta_preds, raw_features):
        x = torch.cat([meta_preds, raw_features], dim=1)
        x = self.batch_norm(x)
        return self.kan3(self.kan2(self.kan1(x))).squeeze()

class AdaptiveBasisFunctions:
    @staticmethod
    def chebyshev_basis(x, max_degree=5):
        basis = [torch.ones_like(x), x]
        for n in range(2, max_degree + 1):
            basis.append(2 * x * basis[-1] - basis[-2])
        return torch.stack(basis, dim=-1)
    @staticmethod
    def fourier_basis(x, n_frequencies=5):
        basis = []
        for k in range(1, n_frequencies + 1):
            basis.extend([torch.sin(k * np.pi * x), torch.cos(k * np.pi * x)])
        return torch.stack(basis, dim=-1)
    @staticmethod
    def gaussian_rbf_basis(x, n_centers=5, sigma=0.3):
        centers = torch.linspace(-1, 1, n_centers, device=x.device)
        return torch.stack([torch.exp(-((x - c) ** 2) / (2 * sigma ** 2)) for c in centers], dim=-1)

class TheoreticalKANLayer(nn.Module):
    def __init__(self, in_features, out_features, basis_config=None):
        super().__init__()
        self.in_features = in_features
        basis_config = basis_config or {'chebyshev_degree': 6, 'fourier_freq': 4, 'rbf_centers': 6}
        self.basis_config = basis_config
        self.total_basis_dim = (basis_config['chebyshev_degree'] + 1) + (2 * basis_config['fourier_freq']) + basis_config['rbf_centers']
        self.basis_weights = nn.Parameter(torch.randn(out_features, in_features, self.total_basis_dim) * 0.01)
        self.connection_weights = nn.Parameter(torch.randn(out_features, in_features) * np.sqrt(2.0 / in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.layer_norm = nn.LayerNorm(out_features)
    def compute_multi_basis(self, x):
        x_min, x_max = x.min(dim=0, keepdim=True)[0], x.max(dim=0, keepdim=True)[0]
        x_norm = 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1
        basis_outputs = []
        for i in range(self.in_features):
            xi = x_norm[:, i:i+1]
            c = AdaptiveBasisFunctions.chebyshev_basis(xi, self.basis_config['chebyshev_degree'])
            f = AdaptiveBasisFunctions.fourier_basis(xi, self.basis_config['fourier_freq'])
            r = AdaptiveBasisFunctions.gaussian_rbf_basis(xi, self.basis_config['rbf_centers'])
            basis_outputs.append(torch.cat([c, f, r], dim=-1).squeeze(1))
        return torch.stack(basis_outputs, dim=1)
    def forward(self, x):
        basis_features = self.compute_multi_basis(x)
        weighted = torch.einsum('bik,oik->boi', basis_features, self.basis_weights)
        output = torch.einsum('boi,oi->bo', weighted, self.connection_weights) + self.bias
        return self.layer_norm(output)
    def compute_sobolev_regularization(self):
        return 1e-4 * (torch.sum(self.basis_weights ** 2) + 0.5 * torch.sum(torch.diff(self.basis_weights, dim=-1)**2))

class HierarchicalKANFusion(nn.Module):
    def __init__(self, num_base_models, input_dim, hidden_dims=[128, 64, 32],
                  basis_config=None, dropout_rate=0.1, uncertainty_estimation=False):
        super().__init__()
        self.uncertainty_estimation = uncertainty_estimation
        combined_input = num_base_models + input_dim
        self.kan_layer1 = TheoreticalKANLayer(combined_input, hidden_dims[0], basis_config)
        self.kan_layer2 = TheoreticalKANLayer(hidden_dims[0], hidden_dims[1], basis_config)
        self.kan_layer3 = TheoreticalKANLayer(hidden_dims[1], hidden_dims[2], basis_config)
        self.dropout = nn.Dropout(dropout_rate)
        if uncertainty_estimation:
            self.mu_head = nn.Linear(hidden_dims[2], 1)
            self.sigma_head = nn.Sequential(nn.Linear(hidden_dims[2], 1), nn.Softplus())
        else:
            self.prediction_head = nn.Linear(hidden_dims[2], 1)
    def forward(self, base_predictions, raw_features):
        x = torch.cat([base_predictions, raw_features], dim=1)
        h1 = self.dropout(self.kan_layer1(x))
        h2 = self.dropout(self.kan_layer2(h1))
        h3 = self.kan_layer3(h2)
        if self.uncertainty_estimation:
            return self.mu_head(h3).squeeze(), self.sigma_head(h3).squeeze() + 1e-6
        return self.prediction_head(h3).squeeze()
    def predict(self, base_predictions, raw_features):
        if self.uncertainty_estimation:
            mu, _ = self.forward(base_predictions, raw_features)
            return mu
        return self.forward(base_predictions, raw_features)
    def compute_total_regularization(self):
        return self.kan_layer1.compute_sobolev_regularization() + \
                self.kan_layer2.compute_sobolev_regularization() + \
                self.kan_layer3.compute_sobolev_regularization()

class CrossCovarianceAttention(nn.Module):
    def __init__(self, num_models, input_dim):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, 64)
        self.key_proj = nn.Linear(num_models, 64)
        self.covariance_proj = nn.Bilinear(input_dim, num_models, 64)
        self.output_head = nn.Sequential(nn.Linear(64 + num_models, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, meta_preds, raw_features):
        Q = self.query_proj(raw_features).unsqueeze(1)
        K = self.key_proj(meta_preds).unsqueeze(1)
        cov_embedding = self.covariance_proj(raw_features, meta_preds)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / 8.0
        attn_probs = torch.sigmoid(attn_scores)
        weighted_cov = cov_embedding * attn_probs.squeeze(-1)
        combined = torch.cat([weighted_cov, meta_preds], dim=1)
        return self.output_head(combined).squeeze()

class DeepEvidentialRegression(nn.Module):
    def __init__(self, num_models, input_dim):
        super().__init__()
        self.input_processor = nn.Sequential(
            nn.Linear(input_dim + num_models, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.GELU())
        self.gamma = nn.Linear(64, 1)
        self.v = nn.Linear(64, 1)
        self.alpha = nn.Linear(64, 1)
        self.beta = nn.Linear(64, 1)
    def forward(self, meta_preds, raw_features):
        x = torch.cat([raw_features, meta_preds], dim=1)
        h = self.input_processor(x)
        gamma = self.gamma(h)
        v = F.softplus(self.v(h)) + 1e-6
        alpha = F.softplus(self.alpha(h)) + 1.0
        beta = F.softplus(self.beta(h)) + 1e-6
        return gamma.squeeze(), v.squeeze(), alpha.squeeze(), beta.squeeze()
    def predict(self, meta_preds, raw_features):
        gamma, _, _, _ = self.forward(meta_preds, raw_features)
        return gamma
    def nig_loss(self, gamma, v, alpha, beta, y):
        two_blambda = 2 * beta * (1 + v)
        nll = 0.5 * torch.log(np.pi / v) - alpha * torch.log(two_blambda) + \
              (alpha + 0.5) * torch.log(v * (y - gamma)**2 + two_blambda) + \
              torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        return nll.mean()

class ImprovedCNN(nn.Module):
    def __init__(self, input_size, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_size)
            x = self.pool(F.relu(self.bn1(self.conv1(dummy))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            self.flatten_dim = x.view(1, -1).shape[1]
        self.fc = nn.Sequential(nn.Linear(self.flatten_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze()

class ImprovedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size * 2, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.fc(out[:, -1, :]).squeeze()

class ImprovedTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, input_size, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Sequential(nn.Linear(d_model * input_size, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        x = self.embedding(x.unsqueeze(-1)) + self.pos_encoder
        x = self.transformer(x)
        x = x.flatten(start_dim=1)
        return self.fc(x).squeeze()

# ==================== YOUR ORIGINAL TRAINERS - NO CHANGES ====================

class BaselineTrainer:
    def __init__(self, model, lr=0.001):
        self.model = model.to(DEVICE)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=10)
        self.best_loss = float('inf')
        self.best_state = None
    def fit(self, train_loader, val_loader):
        patience_counter = 0
        for epoch in range(EPOCHS):
            self.model.train()
            for X, y in train_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X), y)
                loss.backward()
                self.optimizer.step()
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X, y in val_loader:
                    val_loss += self.criterion(self.model(X.to(DEVICE)), y.to(DEVICE)).item()
            val_loss /= len(val_loader)
            self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE: break
        if self.best_state: self.model.load_state_dict(self.best_state)
    def predict(self, X):
        self.model.eval()
        with torch.no_grad(): return self.model(torch.FloatTensor(X).to(DEVICE)).cpu().numpy()

class EnsembleTrainer:
    def __init__(self, model, model_type='standard', lr=0.002):
        self.model = model.to(DEVICE); self.model_type = model_type
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=15)
        self.best_loss = float('inf'); self.best_state = None
    def fit(self, train_loader, val_loader):
        patience_counter = 0
        for epoch in range(EPOCHS):
            self.model.train()
            for meta_pred, raw_feat, y in train_loader:
                meta_pred, raw_feat, y = meta_pred.to(DEVICE), raw_feat.to(DEVICE), y.to(DEVICE)
                self.optimizer.zero_grad()
                if self.model_type == 'evidential':
                    gamma, v, alpha, beta = self.model(meta_pred, raw_feat)
                    loss = self.model.nig_loss(gamma, v, alpha, beta, y)
                else:
                    pred = self.model(meta_pred, raw_feat)
                    loss = self.criterion(pred, y)
                loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); self.optimizer.step()
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for meta_pred, raw_feat, y in val_loader:
                    meta_pred, raw_feat, y = meta_pred.to(DEVICE), raw_feat.to(DEVICE), y.to(DEVICE)
                    pred = self.model.predict(meta_pred, raw_feat) if self.model_type == 'evidential' else self.model(meta_pred, raw_feat)
                    val_loss += self.criterion(pred, y).item()
            val_loss /= len(val_loader); self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE: break
        if self.best_state: self.model.load_state_dict(self.best_state)
    def predict(self, meta_preds, raw_features):
        self.model.eval()
        with torch.no_grad():
            t_meta = torch.FloatTensor(meta_preds).to(DEVICE)
            t_raw = torch.FloatTensor(raw_features).to(DEVICE)
            res = self.model.predict(t_meta, t_raw) if self.model_type == 'evidential' else self.model(t_meta, t_raw)
            return res.cpu().numpy()

class TheoreticalKANTrainer:
    def __init__(self, model, lr=0.001):
        self.model = model.to(DEVICE)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.best_loss = float('inf'); self.best_state = None
    def fit(self, train_loader, val_loader):
        p_c = 0
        for epoch in range(EPOCHS):
            self.model.train()
            for m, r, y in train_loader:
                m, r, y = m.to(DEVICE), r.to(DEVICE), y.to(DEVICE); self.optimizer.zero_grad()
                if self.model.uncertainty_estimation:
                    mu, sig = self.model(m, r)
                    mse_loss = torch.mean(0.5 * torch.log(sig) + 0.5 * ((mu - y)**2) / sig)
                else: mse_loss = self.criterion(self.model(m, r), y)
                loss = mse_loss + 0.01 * self.model.compute_total_regularization()
                loss.backward(); self.optimizer.step()
            self.model.eval()
            with torch.no_grad():
                val_l = sum(self.criterion(self.model.predict(m.to(DEVICE), r.to(DEVICE)), y.to(DEVICE)).item() for m, r, y in val_loader) / len(val_loader)
            if val_l < self.best_loss:
                self.best_loss, self.best_state = val_l, self.model.state_dict().copy(); p_c = 0
            else:
                p_c += 1
                if p_c >= PATIENCE: break
        if self.best_state: self.model.load_state_dict(self.best_state)
    def predict(self, m, r):
        self.model.eval()
        with torch.no_grad():
            return self.model.predict(torch.FloatTensor(m).to(DEVICE), torch.FloatTensor(r).to(DEVICE)).cpu().numpy()
    def predict_with_uncertainty(self, m, r):
        self.model.eval()
        with torch.no_grad():
            t_m = torch.FloatTensor(m).to(DEVICE)
            t_r = torch.FloatTensor(r).to(DEVICE)
            mu, sigma = self.model(t_m, t_r)
            return mu.cpu().numpy(), sigma.cpu().numpy()

# ==================== SHAP COMPUTATION UTILITIES ====================

class SHAPComputer:
    """Compute SHAP values for all models"""
    
    @staticmethod
    def compute_shap_for_ml_model(model, X_background, X_test, feature_names, model_name, seed):
        """Compute SHAP for sklearn-compatible models"""
        try:
            # Use TreeExplainer for tree-based models
            if isinstance(model, (RandomForestRegressor, xgb.XGBRegressor, lgb.LGBMRegressor)):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
            else:
                # Use KernelExplainer for other models (e.g., SVR)
                explainer = shap.KernelExplainer(model.predict, X_background[:100])  # Sample for efficiency
                shap_values = explainer.shap_values(X_test[:100])  # Limit test samples for SVR
            
            return shap_values
        except Exception as e:
            print(f"      Warning: SHAP computation failed for {model_name}: {e}")
            return None
    
    @staticmethod
    def compute_shap_for_dl_model(predict_fn, X_background, X_test, feature_names, model_name, seed):
        """Compute SHAP for deep learning models"""
        try:
            # Wrapper to ensure predict_fn returns 1D array
            def predict_wrapper(X):
                preds = predict_fn(X)
                if isinstance(preds, np.ndarray):
                    if preds.ndim == 0:  # scalar
                        return np.array([preds])
                    elif preds.ndim == 1:
                        return preds
                    else:
                        return preds.flatten()
                else:
                    return np.array([preds])
            
            # Use KernelExplainer for DL models
            explainer = shap.KernelExplainer(predict_wrapper, X_background[:50])  # Sample for efficiency
            shap_values = explainer.shap_values(X_test[:50])  # Limit test samples
            
            return shap_values
        except Exception as e:
            print(f"      Warning: SHAP computation failed for {model_name}: {e}")
            return None
    
    @staticmethod
    def save_shap_values(shap_values, feature_names, model_name, seed, output_dir):
        """Save SHAP values to disk"""
        if shap_values is None:
            return
        
        seed_dir = f"{output_dir}/shap/seed_{seed}"
        os.makedirs(seed_dir, exist_ok=True)
        
        # Save raw SHAP values
        np.save(f"{seed_dir}/{model_name}_shap_values.npy", shap_values)
        
        # Compute and save feature importance (mean absolute SHAP)
        importance = np.mean(np.abs(shap_values), axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv(f"{seed_dir}/{model_name}_feature_importance.csv", index=False)

# ==================== MAIN TRAINING FRAMEWORK WITH RUNTIME & SHAP ====================

class TrainingFramework:
    def __init__(self):
        self.feature_names = ['area_cm2', 'panjang_cm', 'lebar_cm', 'jarak_kamera_cm', 
                              'ketebalan_cm', 'aspect_ratio', 'perimeter_cm', 'solidity']
        self.detailed_results = []
        self.runtime_metrics = []  # NEW: Track runtime
        
    def load_data(self):
        print("\n--- Loading Data ---")
        try:
            df_feat = pd.read_csv(FEATURES_CSV)
            if len(df_feat.columns) == 1 and ';' in str(df_feat.columns[0]): 
                df_feat = pd.read_csv(FEATURES_CSV, sep=';')
            df_gt = pd.read_excel(GROUND_TRUTH_FILE) if GROUND_TRUTH_FILE.endswith('.xlsx') else pd.read_csv(GROUND_TRUTH_FILE)
            if len(df_gt.columns) == 1 and ';' in str(df_gt.columns[0]): 
                df_gt = pd.read_csv(GROUND_TRUTH_FILE, sep=';')
            df_feat.columns = [str(c).strip().lower() for c in df_feat.columns]
            df_gt.columns = [str(c).strip().lower() for c in df_gt.columns]
            rename_map = {'filename': 'image_file', 'berat_kering': 'berat_kering_gram', 'object': 'object_id'}
            df_feat.rename(columns=rename_map, inplace=True); df_gt.rename(columns=rename_map, inplace=True)
            df = pd.merge(df_feat, df_gt, on=['image_file', 'object_id'], how='inner')
            valid_feats = [f for f in self.feature_names if f in df.columns]
            self.feature_names = valid_feats
            df = df[(np.abs(stats.zscore(df['berat_kering_gram'])) < OUTLIER_THRESHOLD)]
            self.X_all, self.y_all = df[self.feature_names].values, df['berat_kering_gram'].values
            print(f"✅ Data Loaded: {len(df)} samples, {len(self.feature_names)} features")
            return True
        except Exception as e: 
            print(f"❌ ERROR: {e}")
            return False

    def run_training(self):
        """EXACT SAME TRAINING LOOP AS YOUR ORIGINAL CODE WITH RUNTIME TRACKING"""
        print("\n" + "="*70)
        print("  PART 1: MODEL TRAINING & SAVING (WITH RUNTIME & SHAP)")
        print("="*70)
        
        for i, seed in enumerate(SEEDS):
            print(f"\n{'='*70}")
            print(f"  ITERATION {i+1}/15 (Seed: {seed})")
            print(f"{'='*70}")
            start_time = time.time()
            self._run_iteration(seed, i+1)
            elapsed = (time.time() - start_time) / 60
            print(f"  ✓ Iteration completed in {elapsed:.2f} minutes")
        
        # Save results
        df = pd.DataFrame(self.detailed_results)
        df.to_csv(f"{OUTPUT_DIR}/tables/raw_results.csv", index=False)
        
        # NEW: Save runtime metrics
        runtime_df = pd.DataFrame(self.runtime_metrics)
        runtime_df.to_csv(f"{OUTPUT_DIR}/tables/runtime_metrics.csv", index=False)
        
        print(f"\n✅ Training Complete!")
        print(f"   Results: {OUTPUT_DIR}/tables/raw_results.csv")
        print(f"   Runtime: {OUTPUT_DIR}/tables/runtime_metrics.csv")

    def _run_iteration(self, seed, iter_num):
        """YOUR EXACT ORIGINAL TRAINING LOGIC WITH RUNTIME & SHAP TRACKING"""
        X_temp, X_test, y_temp, y_test = train_test_split(self.X_all, self.y_all, test_size=TEST_SIZE, random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=VALIDATION_SIZE/(1-TEST_SIZE), random_state=seed)
        scaler = StandardScaler()
        X_tr_s, X_vl_s, X_te_s = scaler.fit_transform(X_train), scaler.transform(X_val), scaler.transform(X_test)
        
        # Save data splits for evaluation
        np.save(f"{OUTPUT_DIR}/predictions/seed_{seed}_y_test.npy", y_test)
        np.save(f"{OUTPUT_DIR}/predictions/seed_{seed}_y_train.npy", y_train)
        np.save(f"{OUTPUT_DIR}/predictions/seed_{seed}_X_test.npy", X_te_s)  # NEW: Save for SHAP
        with open(f"{OUTPUT_DIR}/models/seed_{seed}_scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        
        # Baselines ML
        ml_models = {}
        print("  - Training ML Baselines...")
        for name, mod in [('RF', RandomForestRegressor(n_estimators=200, random_state=seed)), 
                          ('XGB', xgb.XGBRegressor(n_estimators=200, random_state=seed, verbosity=0)),
                          ('LGBM', lgb.LGBMRegressor(n_estimators=200, random_state=seed, verbose=-1)),
                          ('SVR', SVR(C=10))]:
            # Training time
            train_start = time.time()
            mod.fit(X_tr_s, y_train)
            train_time = time.time() - train_start
            
            ml_models[name] = mod
            
            # Inference time
            infer_start = time.time()
            train_pred = mod.predict(X_tr_s)
            test_pred = mod.predict(X_te_s)
            infer_time = time.time() - infer_start
            
            self._log(iter_num, f"Base_{name}", y_train, train_pred, y_test, test_pred, seed)
            
            # NEW: Log runtime
            self.runtime_metrics.append({
                'Seed': seed,
                'Model': f"Base_{name}",
                'Train_Time_sec': train_time,
                'Inference_Time_sec': infer_time,
                'Total_Time_sec': train_time + infer_time
            })
            
            # NEW: Compute SHAP
            shap_values = SHAPComputer.compute_shap_for_ml_model(
                mod, X_tr_s[:100], X_te_s, self.feature_names, f"Base_{name}", seed)
            SHAPComputer.save_shap_values(shap_values, self.feature_names, f"Base_{name}", seed, OUTPUT_DIR)

        # Baselines DL
        print("  - Training DL Baselines...")
        tr_l = DataLoader(TensorDataset(torch.FloatTensor(X_tr_s), torch.FloatTensor(y_train)), BATCH_SIZE, shuffle=True)
        vl_l = DataLoader(TensorDataset(torch.FloatTensor(X_vl_s), torch.FloatTensor(y_val)), BATCH_SIZE)
        dl_models = {}
        for name, cls in [('CNN', ImprovedCNN), ('LSTM', ImprovedLSTM), ('Transformer', ImprovedTransformer)]:
            # Training time
            train_start = time.time()
            t = BaselineTrainer(cls(X_tr_s.shape[1]))
            t.fit(tr_l, vl_l)
            train_time = time.time() - train_start
            
            dl_models[name] = t
            
            # Inference time
            infer_start = time.time()
            train_pred = t.predict(X_tr_s)
            test_pred = t.predict(X_te_s)
            infer_time = time.time() - infer_start
            
            self._log(iter_num, f"Base_{name}", y_train, train_pred, y_test, test_pred, seed)
            
            # NEW: Log runtime
            self.runtime_metrics.append({
                'Seed': seed,
                'Model': f"Base_{name}",
                'Train_Time_sec': train_time,
                'Inference_Time_sec': infer_time,
                'Total_Time_sec': train_time + infer_time
            })
            
            # NEW: Compute SHAP for DL
            shap_values = SHAPComputer.compute_shap_for_dl_model(
                t.predict, X_tr_s[:50], X_te_s[:50], self.feature_names, f"Base_{name}", seed)
            SHAPComputer.save_shap_values(shap_values, self.feature_names, f"Base_{name}", seed, OUTPUT_DIR)

        # Meta-Features
        m_tr = np.column_stack([m.predict(X_tr_s) for m in ml_models.values()] + [m.predict(X_tr_s) for m in dl_models.values()])
        m_te = np.column_stack([m.predict(X_te_s) for m in ml_models.values()] + [m.predict(X_te_s) for m in dl_models.values()])
        m_vl = np.column_stack([m.predict(X_vl_s) for m in ml_models.values()] + [m.predict(X_vl_s) for m in dl_models.values()])
        h_tr_l = DataLoader(TensorDataset(torch.FloatTensor(m_tr), torch.FloatTensor(X_tr_s), torch.FloatTensor(y_train)), BATCH_SIZE, shuffle=True)
        h_vl_l = DataLoader(TensorDataset(torch.FloatTensor(m_vl), torch.FloatTensor(X_vl_s), torch.FloatTensor(y_val)), BATCH_SIZE)
        num_m, in_d = m_tr.shape[1], X_tr_s.shape[1]
        
        # Hybrids
        print("  - Training Novel Hybrids...")
        for name, mod in [('DGFC', DynamicGraphFeatureConvolution(num_m, in_d)),
                          ('RECN', ResidualErrorCorrectingNetwork(num_m, in_d)),
                          ('KAN_Std', KANFusionNetwork(num_m, in_d)),
                          ('CCAN', CrossCovarianceAttention(num_m, in_d))]:
            # Training time
            train_start = time.time()
            t = EnsembleTrainer(mod)
            t.fit(h_tr_l, h_vl_l)
            train_time = time.time() - train_start
            
            # Inference time
            infer_start = time.time()
            train_pred = t.predict(m_tr, X_tr_s)
            test_pred = t.predict(m_te, X_te_s)
            infer_time = time.time() - infer_start
            
            self._log(iter_num, f"Novel_{name}", y_train, train_pred, y_test, test_pred, seed)
            
            # NEW: Log runtime
            self.runtime_metrics.append({
                'Seed': seed,
                'Model': f"Novel_{name}",
                'Train_Time_sec': train_time,
                'Inference_Time_sec': infer_time,
                'Total_Time_sec': train_time + infer_time
            })
            
            # NEW: Compute SHAP (using wrapper for hybrid input)
            def hybrid_predict(X):
                m_pred = np.column_stack([ml_models[mn].predict(X) for mn in ['RF', 'XGB', 'LGBM', 'SVR']] + 
                                        [dl_models[dn].predict(X) for dn in ['CNN', 'LSTM', 'Transformer']])
                preds = t.predict(m_pred, X)
                # Ensure we return a 1D array
                if isinstance(preds, np.ndarray):
                    if preds.ndim == 0:
                        return np.array([preds])
                    else:
                        return preds.flatten()
                else:
                    return np.array([preds])
            
            shap_values = SHAPComputer.compute_shap_for_dl_model(
                hybrid_predict, X_tr_s[:50], X_te_s[:50], self.feature_names, f"Novel_{name}", seed)
            SHAPComputer.save_shap_values(shap_values, self.feature_names, f"Novel_{name}", seed, OUTPUT_DIR)
            
        # DERH
        train_start = time.time()
        derh_t = EnsembleTrainer(DeepEvidentialRegression(num_m, in_d), 'evidential')
        derh_t.fit(h_tr_l, h_vl_l)
        train_time = time.time() - train_start
        
        infer_start = time.time()
        train_pred = derh_t.predict(m_tr, X_tr_s)
        test_pred = derh_t.predict(m_te, X_te_s)
        infer_time = time.time() - infer_start
        
        self._log(iter_num, "Novel_DERH", y_train, train_pred, y_test, test_pred, seed)
        
        self.runtime_metrics.append({
            'Seed': seed,
            'Model': "Novel_DERH",
            'Train_Time_sec': train_time,
            'Inference_Time_sec': infer_time,
            'Total_Time_sec': train_time + infer_time
        })
        
        def derh_predict(X):
            m_pred = np.column_stack([ml_models[mn].predict(X) for mn in ['RF', 'XGB', 'LGBM', 'SVR']] + 
                                    [dl_models[dn].predict(X) for dn in ['CNN', 'LSTM', 'Transformer']])
            preds = derh_t.predict(m_pred, X)
            # Ensure we return a 1D array
            if isinstance(preds, np.ndarray):
                if preds.ndim == 0:
                    return np.array([preds])
                else:
                    return preds.flatten()
            else:
                return np.array([preds])
        
        shap_values = SHAPComputer.compute_shap_for_dl_model(
            derh_predict, X_tr_s[:50], X_te_s[:50], self.feature_names, "Novel_DERH", seed)
        SHAPComputer.save_shap_values(shap_values, self.feature_names, "Novel_DERH", seed, OUTPUT_DIR)

        # Advanced KAN Variants
        for name, cfg, unc in [('KAN_HighOrder', {'chebyshev_degree': 10, 'fourier_freq': 6, 'rbf_centers': 10}, False),
                                ('KAN_Uncertainty', None, True)]:
            train_start = time.time()
            kan_v = HierarchicalKANFusion(num_m, in_d, basis_config=cfg, uncertainty_estimation=unc)
            t_kan = TheoreticalKANTrainer(kan_v)
            t_kan.fit(h_tr_l, h_vl_l)
            train_time = time.time() - train_start
            
            infer_start = time.time()
            train_pred = t_kan.predict(m_tr, X_tr_s)
            test_pred = t_kan.predict(m_te, X_te_s)
            infer_time = time.time() - infer_start
            
            # Save uncertainty if applicable
            if unc:
                test_pred_mu, test_sigma = t_kan.predict_with_uncertainty(m_te, X_te_s)
                np.save(f"{OUTPUT_DIR}/predictions/seed_{seed}_Novel_{name}_pred.npy", test_pred_mu)
                np.save(f"{OUTPUT_DIR}/predictions/seed_{seed}_Novel_{name}_std.npy", test_sigma)
            
            self._log(iter_num, f"Novel_{name}", y_train, train_pred, y_test, test_pred, seed)
            
            self.runtime_metrics.append({
                'Seed': seed,
                'Model': f"Novel_{name}",
                'Train_Time_sec': train_time,
                'Inference_Time_sec': infer_time,
                'Total_Time_sec': train_time + infer_time
            })
            
            def kan_predict(X):
                m_pred = np.column_stack([ml_models[mn].predict(X) for mn in ['RF', 'XGB', 'LGBM', 'SVR']] + 
                                        [dl_models[dn].predict(X) for dn in ['CNN', 'LSTM', 'Transformer']])
                preds = t_kan.predict(m_pred, X)
                # Ensure we return a 1D array
                if isinstance(preds, np.ndarray):
                    if preds.ndim == 0:
                        return np.array([preds])
                    else:
                        return preds.flatten()
                else:
                    return np.array([preds])
            
            shap_values = SHAPComputer.compute_shap_for_dl_model(
                kan_predict, X_tr_s[:50], X_te_s[:50], self.feature_names, f"Novel_{name}", seed)
            SHAPComputer.save_shap_values(shap_values, self.feature_names, f"Novel_{name}", seed, OUTPUT_DIR)

    def _log(self, it, mod, y_train, train_pred, y_test, test_pred, seed):
        """YOUR ORIGINAL LOGGING + Save predictions"""
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        mae = mean_absolute_error(y_test, test_pred)
        
        print(f"      > {mod:20} | R2: {test_r2:.4f}, MAE: {mae:.4f}")
        
        self.detailed_results.append({
            'Iteration': it, 
            'Model': mod, 
            'R2': test_r2, 
            'MAE': mae,
            'Train_R2': train_r2,
            'Seed': seed
        })
        
        # Save predictions for later evaluation
        np.save(f"{OUTPUT_DIR}/predictions/seed_{seed}_{mod}_pred.npy", test_pred)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  EAAI FRAMEWORK - PART 1: TRAINING (ENHANCED)")
    print("="*70 + "\n")
    
    framework = TrainingFramework()
    if framework.load_data():
        framework.run_training()
        
        print("\n" + "="*70)
        print("  ✅ TRAINING COMPLETE!")
        print("  📁 Models saved to: " + OUTPUT_DIR + "/models/")
        print("  📁 Predictions saved to: " + OUTPUT_DIR + "/predictions/")
        print("  📁 Raw results: " + OUTPUT_DIR + "/tables/raw_results.csv")
        print("  📁 Runtime metrics: " + OUTPUT_DIR + "/tables/runtime_metrics.csv")
        print("  📁 SHAP values: " + OUTPUT_DIR + "/shap/")
        print("\n  ➡️  Now run PART 2 for comprehensive evaluation!")
        print("="*70 + "\n")