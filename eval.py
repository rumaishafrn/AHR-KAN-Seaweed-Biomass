import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon, norm, spearmanr
import scikit_posthocs as sp
import warnings
import os
import json
import glob

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ======================= CONFIGURATION =======================
OUTPUT_DIR = "EAAI_journal_results_complete_FINAL"  # Must match Part 1 output directory

# Create evaluation subdirectories
for subdir in ['visualizations', 'statistical_tests', 'stability', 'uncertainty', 'comparison',
               'spearman', 'runtime', 'shap_analysis', 'feature_correlation', 'ablation', 'failure_analysis']:
    os.makedirs(f"{OUTPUT_DIR}/{subdir}", exist_ok=True)

SEEDS = [42, 123, 7, 2024, 88, 999, 101, 55, 303, 777, 1337, 2026, 11, 888, 555]
FEATURE_NAMES = ['area_cm2', 'panjang_cm', 'lebar_cm', 'jarak_kamera_cm', 
                 'ketebalan_cm', 'aspect_ratio', 'perimeter_cm', 'solidity']

# ==================== EVALUATION CLASSES ====================

class ComprehensiveEvaluator:
    """All evaluation metrics"""
    
    @staticmethod
    def compute_all_metrics(y_true, y_pred, model_params=10000):
        n = len(y_true)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        rss = np.sum((y_true - y_pred) ** 2)
        k = model_params
        aic = n * np.log(rss / n) + 2 * k if rss > 0 else np.inf
        bic = n * np.log(rss / n) + k * np.log(n) if rss > 0 else np.inf
        return {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'AIC': aic, 'BIC': bic}
    
    @staticmethod
    def generalization_gap(train_r2, test_r2):
        gap = train_r2 - test_r2
        if gap > 0.10: status = "Overfitting"
        elif gap < -0.05: status = "Underfitting"
        else: status = "Well-Generalized"
        return gap, status
    
    @staticmethod
    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        if abs(d) < 0.2: interpretation = "Small"
        elif abs(d) < 0.8: interpretation = "Medium"
        else: interpretation = "Large"
        return d, interpretation

class UncertaintyEvaluator:
    """Uncertainty metrics for probabilistic models"""
    
    @staticmethod
    def prediction_interval_coverage(y_true, y_pred, y_std, confidence=0.95):
        z_score = norm.ppf((1 + confidence) / 2)
        lower = y_pred - z_score * y_std
        upper = y_pred + z_score * y_std
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        return coverage
    
    @staticmethod
    def mean_prediction_interval_width(y_std, confidence=0.95):
        z_score = norm.ppf((1 + confidence) / 2)
        mpiw = np.mean(2 * z_score * y_std)
        return mpiw
    
    @staticmethod
    def coverage_width_criterion(y_true, y_pred, y_std, confidence=0.95, eta=50):
        picp = UncertaintyEvaluator.prediction_interval_coverage(y_true, y_pred, y_std, confidence)
        mpiw = UncertaintyEvaluator.mean_prediction_interval_width(y_std, confidence)
        penalty = 1 + np.exp(-eta * (picp - confidence))
        cwc = mpiw * penalty
        return cwc, picp, mpiw

# ==================== NEW: SPEARMAN CORRELATION ANALYZER ====================

class SpearmanAnalyzer:
    """Compute Spearman rank correlation for all models"""
    
    @staticmethod
    def compute_spearman_for_all_models(prediction_data):
        """Compute Spearman correlation coefficient for each model"""
        print("\n  Computing Spearman correlations...")
        
        spearman_results = []
        
        for key, pred_data in prediction_data.items():
            y_true = pred_data['y_true']
            y_pred = pred_data['y_pred']
            
            # Compute Spearman correlation
            rho, p_value = spearmanr(y_true, y_pred)
            
            spearman_results.append({
                'Model': pred_data['model'],
                'Seed': pred_data['seed'],
                'Spearman_Rho': rho,
                'P_Value': p_value,
                'Significant': p_value < 0.05
            })
        
        df = pd.DataFrame(spearman_results)
        return df
    
    @staticmethod
    def generate_spearman_report(spearman_df, output_dir):
        """Generate comprehensive Spearman analysis report"""
        # Summary statistics
        summary = spearman_df.groupby('Model')['Spearman_Rho'].agg(['mean', 'std', 'min', 'max']).round(4)
        summary = summary.sort_values('mean', ascending=False)
        summary.to_csv(f"{output_dir}/spearman/spearman_summary.csv")
        
        print("\n  TOP 10 MODELS BY SPEARMAN CORRELATION:")
        print(summary.head(10).to_string())
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # Box plot
        model_order = spearman_df.groupby('Model')['Spearman_Rho'].median().sort_values(ascending=False).index
        sns.boxplot(data=spearman_df, x='Model', y='Spearman_Rho', order=model_order, palette="Set3", ax=ax1)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.set_title('Spearman Correlation Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Spearman ρ', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='ρ=0.9 (Excellent)')
        ax1.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='ρ=0.7 (Good)')
        ax1.legend()
        
        # Bar plot with error bars
        summary_sorted = summary.sort_values('mean', ascending=False)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(summary_sorted)))
        ax2.barh(range(len(summary_sorted)), summary_sorted['mean'], 
                xerr=summary_sorted['std'], capsize=5, color=colors, 
                edgecolor='black', linewidth=1.5, alpha=0.8)
        ax2.set_yticks(range(len(summary_sorted)))
        ax2.set_yticklabels(summary_sorted.index, fontsize=9)
        ax2.set_xlabel('Mean Spearman ρ', fontsize=12, fontweight='bold')
        ax2.set_title('Average Spearman Correlation (with Std Dev)', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/spearman/spearman_analysis.svg", format='svg', bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Spearman analysis saved to: {output_dir}/spearman/")
        
        return summary

# ==================== NEW: RUNTIME ANALYZER ====================

class RuntimeAnalyzer:
    """Analyze and visualize model runtime"""
    
    @staticmethod
    def analyze_runtime(runtime_df, output_dir):
        """Generate comprehensive runtime analysis"""
        print("\n  Analyzing runtime performance...")
        
        # Summary statistics
        summary = runtime_df.groupby('Model')[['Train_Time_sec', 'Inference_Time_sec', 'Total_Time_sec']].agg(['mean', 'std']).round(3)
        summary.to_csv(f"{output_dir}/runtime/runtime_summary.csv")
        
        print("\n  RUNTIME SUMMARY (seconds):")
        print(summary.head(10).to_string())
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # 1. Training time
        ax1 = axes[0, 0]
        train_summary = runtime_df.groupby('Model')['Train_Time_sec'].mean().sort_values()
        colors = ['green' if x < 10 else 'orange' if x < 30 else 'red' for x in train_summary.values]
        ax1.barh(range(len(train_summary)), train_summary.values, color=colors, 
                edgecolor='black', linewidth=1.5, alpha=0.8)
        ax1.set_yticks(range(len(train_summary)))
        ax1.set_yticklabels(train_summary.index, fontsize=9)
        ax1.set_xlabel('Training Time (seconds)', fontsize=11, fontweight='bold')
        ax1.set_title('Average Training Time\n(Green<10s, Orange<30s, Red≥30s)', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Inference time
        ax2 = axes[0, 1]
        infer_summary = runtime_df.groupby('Model')['Inference_Time_sec'].mean().sort_values()
        colors = ['green' if x < 0.1 else 'orange' if x < 1 else 'red' for x in infer_summary.values]
        ax2.barh(range(len(infer_summary)), infer_summary.values, color=colors, 
                edgecolor='black', linewidth=1.5, alpha=0.8)
        ax2.set_yticks(range(len(infer_summary)))
        ax2.set_yticklabels(infer_summary.index, fontsize=9)
        ax2.set_xlabel('Inference Time (seconds)', fontsize=11, fontweight='bold')
        ax2.set_title('Average Inference Time\n(Green<0.1s, Orange<1s, Red≥1s)', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Total time comparison
        ax3 = axes[1, 0]
        total_summary = runtime_df.groupby('Model')['Total_Time_sec'].mean().sort_values()
        colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(total_summary)))
        ax3.barh(range(len(total_summary)), total_summary.values, color=colors, 
                edgecolor='black', linewidth=1.5, alpha=0.8)
        ax3.set_yticks(range(len(total_summary)))
        ax3.set_yticklabels(total_summary.index, fontsize=9)
        ax3.set_xlabel('Total Time (seconds)', fontsize=11, fontweight='bold')
        ax3.set_title('Total Runtime (Train + Inference)', fontsize=12, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Efficiency scatter (R2 vs Runtime)
        ax4 = axes[1, 1]
        # Load R2 scores
        results_df = pd.read_csv(f"{output_dir}/tables/complete_metrics.csv")
        r2_mean = results_df.groupby('Model')['R2'].mean()
        runtime_mean = runtime_df.groupby('Model')['Total_Time_sec'].mean()
        
        merged = pd.DataFrame({'R2': r2_mean, 'Runtime': runtime_mean}).reset_index()
        
        for _, row in merged.iterrows():
            color = 'green' if 'Base_' in row['Model'] else 'blue'
            marker = 'o' if 'Base_' in row['Model'] else 's'
            ax4.scatter(row['Runtime'], row['R2'], s=100, c=color, marker=marker, 
                       edgecolors='black', linewidth=1.5, alpha=0.7)
            ax4.annotate(row['Model'].replace('Base_', '').replace('Novel_', ''), 
                        (row['Runtime'], row['R2']), fontsize=7, ha='right')
        
        ax4.set_xlabel('Total Runtime (seconds)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('R² Score', fontsize=11, fontweight='bold')
        ax4.set_title('Efficiency Analysis (R² vs Runtime)\nGreen=Baseline, Blue=Novel', 
                     fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/runtime/runtime_analysis.svg", format='svg', bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Runtime analysis saved to: {output_dir}/runtime/")

# ==================== NEW: SHAP ANALYZER ====================

class SHAPAnalyzer:
    """Aggregate and analyze SHAP values across all models"""
    
    @staticmethod
    def aggregate_shap_values(output_dir, feature_names):
        """Aggregate SHAP values across seeds for each model"""
        print("\n  Aggregating SHAP values...")
        
        shap_summary = {}
        
        # Get all unique models
        all_models = set()
        for seed in SEEDS:
            seed_dir = f"{output_dir}/shap/seed_{seed}"
            if os.path.exists(seed_dir):
                csv_files = glob.glob(f"{seed_dir}/*_feature_importance.csv")
                for f in csv_files:
                    model_name = os.path.basename(f).replace('_feature_importance.csv', '')
                    all_models.add(model_name)
        
        # Aggregate for each model
        for model_name in all_models:
            importances_list = []
            
            for seed in SEEDS:
                importance_file = f"{output_dir}/shap/seed_{seed}/{model_name}_feature_importance.csv"
                if os.path.exists(importance_file):
                    df = pd.read_csv(importance_file)
                    importances_list.append(df.set_index('feature')['importance'])
            
            if importances_list:
                # Average across seeds
                avg_importance = pd.concat(importances_list, axis=1).mean(axis=1)
                std_importance = pd.concat(importances_list, axis=1).std(axis=1)
                
                shap_summary[model_name] = pd.DataFrame({
                    'feature': avg_importance.index,
                    'mean_importance': avg_importance.values,
                    'std_importance': std_importance.values
                }).sort_values('mean_importance', ascending=False)
        
        return shap_summary
    
    @staticmethod
    def visualize_shap_summary(shap_summary, output_dir):
        """Create comprehensive SHAP visualization"""
        print("\n  Creating SHAP visualizations...")
        
        # 1. Individual plots for each model
        for model_name, importance_df in shap_summary.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
            ax.barh(range(len(importance_df)), importance_df['mean_importance'], 
                   xerr=importance_df['std_importance'], capsize=5, color=colors, 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
            ax.set_yticks(range(len(importance_df)))
            ax.set_yticklabels(importance_df['feature'], fontsize=11)
            ax.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
            ax.set_title(f'{model_name}\nFeature Importance (SHAP)', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/shap_analysis/{model_name}_shap.svg", format='svg', bbox_inches='tight')
            plt.close()
        
        # 2. Comparison heatmap (all models, all features)
        all_models_list = sorted(shap_summary.keys())
        feature_names = shap_summary[all_models_list[0]]['feature'].tolist()
        
        heatmap_data = np.zeros((len(all_models_list), len(feature_names)))
        
        for i, model_name in enumerate(all_models_list):
            for j, feature in enumerate(feature_names):
                imp_row = shap_summary[model_name][shap_summary[model_name]['feature'] == feature]
                if not imp_row.empty:
                    heatmap_data[i, j] = imp_row['mean_importance'].values[0]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                   xticklabels=feature_names, yticklabels=all_models_list,
                   cbar_kws={'label': 'Mean |SHAP Value|'}, linewidths=0.5, ax=ax)
        ax.set_title('SHAP Feature Importance Heatmap - All Models', fontsize=15, fontweight='bold')
        ax.set_xlabel('Features', fontsize=12, fontweight='bold')
        ax.set_ylabel('Models', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_analysis/shap_heatmap_all_models.svg", format='svg', bbox_inches='tight')
        plt.close()
        
        # 3. Top features across all models
        all_importances = []
        for model_name, importance_df in shap_summary.items():
            for _, row in importance_df.iterrows():
                all_importances.append({
                    'feature': row['feature'],
                    'importance': row['mean_importance']
                })
        
        overall_importance = pd.DataFrame(all_importances).groupby('feature')['importance'].mean().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(overall_importance)))
        ax.barh(range(len(overall_importance)), overall_importance.values, color=colors, 
               edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.set_yticks(range(len(overall_importance)))
        ax.set_yticklabels(overall_importance.index, fontsize=12, fontweight='bold')
        ax.set_xlabel('Average Importance Across All Models', fontsize=13, fontweight='bold')
        ax.set_title('Overall Feature Importance (SHAP)\nAveraged Across All 14 Models', 
                    fontsize=15, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_analysis/overall_feature_importance.svg", format='svg', bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ SHAP analysis saved to: {output_dir}/shap_analysis/")
        
        return overall_importance

# ==================== NEW: FEATURE CORRELATION ANALYZER ====================

class FeatureCorrelationAnalyzer:
    """Analyze correlation between features and target"""
    
    @staticmethod
    def compute_feature_correlations(output_dir):
        """Compute correlation matrix between features and biomass"""
        print("\n  Computing feature-biomass correlations...")
        
        # Load original data
        X_test_all = []
        y_test_all = []
        
        for seed in SEEDS:
            X_test = np.load(f"{output_dir}/predictions/seed_{seed}_X_test.npy")
            y_test = np.load(f"{output_dir}/predictions/seed_{seed}_y_test.npy")
            X_test_all.append(X_test)
            y_test_all.append(y_test)
        
        # Concatenate all test sets
        X_all = np.vstack(X_test_all)
        y_all = np.concatenate(y_test_all)
        
        # Create dataframe
        df = pd.DataFrame(X_all, columns=FEATURE_NAMES)
        df['biomass'] = y_all
        
        # Compute correlations
        corr_matrix = df.corr()
        
        # Pearson correlation
        pearson_with_biomass = corr_matrix['biomass'].drop('biomass').sort_values(ascending=False)
        
        # Spearman correlation
        spearman_matrix = df.corr(method='spearman')
        spearman_with_biomass = spearman_matrix['biomass'].drop('biomass').sort_values(ascending=False)
        
        # Save results
        correlation_summary = pd.DataFrame({
            'Feature': pearson_with_biomass.index,
            'Pearson': pearson_with_biomass.values,
            'Spearman': spearman_with_biomass.values
        })
        correlation_summary.to_csv(f"{output_dir}/feature_correlation/feature_biomass_correlation.csv", index=False)
        
        print("\n  FEATURE-BIOMASS CORRELATIONS:")
        print(correlation_summary.to_string())
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # Full correlation heatmap
        ax1 = axes[0]
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, cbar_kws={'label': 'Correlation'}, ax=ax1)
        ax1.set_title('Feature Correlation Matrix (Pearson)\nIncluding Biomass Target', 
                     fontsize=14, fontweight='bold')
        
        # Feature-biomass correlation bar plot
        ax2 = axes[1]
        features = correlation_summary['Feature'].tolist()
        x_pos = np.arange(len(features))
        width = 0.35
        
        bars1 = ax2.bar(x_pos - width/2, correlation_summary['Pearson'], width, 
                       label='Pearson', color='steelblue', edgecolor='black', linewidth=1.5, alpha=0.8)
        bars2 = ax2.bar(x_pos + width/2, correlation_summary['Spearman'], width, 
                       label='Spearman', color='orange', edgecolor='black', linewidth=1.5, alpha=0.8)
        
        ax2.set_xlabel('Features', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Correlation Coefficient', fontsize=12, fontweight='bold')
        ax2.set_title('Feature-Biomass Correlation\n(Pearson vs Spearman)', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(features, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.legend(fontsize=11)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_correlation/correlation_analysis.svg", format='svg', bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Feature correlation analysis saved to: {output_dir}/feature_correlation/")
        
        return correlation_summary

# ==================== NEW: ABLATION STUDY ====================

class AblationAnalyzer:
    """Perform feature ablation study"""
    
    @staticmethod
    def analyze_feature_ablation(output_dir):
        """Analyze impact of removing each feature"""
        print("\n  Performing ablation study...")
        
        # This is a post-hoc analysis using SHAP values
        # We estimate the impact by analyzing SHAP importance
        
        ablation_results = []
        
        # Load SHAP summaries
        shap_summary = {}
        all_models = set()
        
        for seed in SEEDS:
            seed_dir = f"{output_dir}/shap/seed_{seed}"
            if os.path.exists(seed_dir):
                csv_files = glob.glob(f"{seed_dir}/*_feature_importance.csv")
                for f in csv_files:
                    model_name = os.path.basename(f).replace('_feature_importance.csv', '')
                    all_models.add(model_name)
        
        # For each model, compute relative importance
        for model_name in sorted(all_models):
            importances_list = []
            
            for seed in SEEDS:
                importance_file = f"{output_dir}/shap/seed_{seed}/{model_name}_feature_importance.csv"
                if os.path.exists(importance_file):
                    df = pd.read_csv(importance_file)
                    importances_list.append(df.set_index('feature')['importance'])
            
            if importances_list:
                avg_importance = pd.concat(importances_list, axis=1).mean(axis=1)
                total_importance = avg_importance.sum()
                
                for feature in FEATURE_NAMES:
                    if feature in avg_importance.index:
                        relative_impact = (avg_importance[feature] / total_importance) * 100
                        ablation_results.append({
                            'Model': model_name,
                            'Feature': feature,
                            'Absolute_Importance': avg_importance[feature],
                            'Relative_Impact_%': relative_impact
                        })
        
        ablation_df = pd.DataFrame(ablation_results)
        ablation_df.to_csv(f"{output_dir}/ablation/ablation_study.csv", index=False)
        
        # Summary: Average impact across all models
        avg_impact = ablation_df.groupby('Feature')['Relative_Impact_%'].mean().sort_values(ascending=False)
        
        print("\n  AVERAGE FEATURE IMPACT (% of total importance):")
        print(avg_impact.to_string())
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # Heatmap: Feature impact per model
        pivot = ablation_df.pivot_table(values='Relative_Impact_%', index='Model', columns='Feature', aggfunc='first')
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Relative Impact (%)'}, linewidths=0.5, ax=ax1)
        ax1.set_title('Feature Ablation Study\nRelative Impact per Model (%)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Features', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Models', fontsize=12, fontweight='bold')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Bar plot: Average impact
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(avg_impact)))
        ax2.barh(range(len(avg_impact)), avg_impact.values, color=colors, 
                edgecolor='black', linewidth=1.5, alpha=0.8)
        ax2.set_yticks(range(len(avg_impact)))
        ax2.set_yticklabels(avg_impact.index, fontsize=11, fontweight='bold')
        ax2.set_xlabel('Average Relative Impact (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Average Feature Impact\nAcross All Models', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()
        
        # Add percentage labels
        for i, v in enumerate(avg_impact.values):
            ax2.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ablation/ablation_analysis.svg", format='svg', bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Ablation study saved to: {output_dir}/ablation/")
        
        return ablation_df

# ==================== NEW: FAILURE MODE ANALYZER ====================

class FailureModeAnalyzer:
    """Identify and analyze failure cases"""
    
    @staticmethod
    def identify_failure_modes(prediction_data, output_dir):
        """Identify worst predictions and analyze why they fail"""
        print("\n  Analyzing failure modes...")
        
        failure_analysis = []
        
        # For each model, find worst predictions
        models = list(set([pred_data['model'] for pred_data in prediction_data.values()]))
        
        for model_name in sorted(models):
            # Collect all predictions for this model across seeds
            all_errors = []
            all_y_true = []
            all_y_pred = []
            all_X_test = []
            
            for seed in SEEDS:
                key = f"{model_name}_seed_{seed}"
                if key in prediction_data:
                    pred_data = prediction_data[key]
                    y_true = pred_data['y_true']
                    y_pred = pred_data['y_pred']
                    
                    errors = np.abs(y_true - y_pred)
                    all_errors.extend(errors)
                    all_y_true.extend(y_true)
                    all_y_pred.extend(y_pred)
                    
                    # Load corresponding X_test
                    X_test = np.load(f"{output_dir}/predictions/seed_{seed}_X_test.npy")
                    all_X_test.extend(X_test)
            
            all_errors = np.array(all_errors)
            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)
            all_X_test = np.array(all_X_test)
            
            # Find worst 10% predictions
            threshold = np.percentile(all_errors, 90)
            worst_indices = np.where(all_errors >= threshold)[0]
            
            # Analyze characteristics of worst predictions
            worst_true_values = all_y_true[worst_indices]
            worst_pred_values = all_y_pred[worst_indices]
            worst_errors = all_errors[worst_indices]
            worst_X = all_X_test[worst_indices]
            
            failure_analysis.append({
                'Model': model_name,
                'Num_Failures': len(worst_indices),
                'Avg_Error': worst_errors.mean(),
                'Max_Error': worst_errors.max(),
                'Avg_True_Value': worst_true_values.mean(),
                'Avg_Pred_Value': worst_pred_values.mean(),
                'True_Value_Range': f"{worst_true_values.min():.2f}-{worst_true_values.max():.2f}",
                # Feature statistics for worst cases
                **{f'Avg_{feat}': worst_X[:, i].mean() for i, feat in enumerate(FEATURE_NAMES)}
            })
        
        failure_df = pd.DataFrame(failure_analysis)
        failure_df = failure_df.sort_values('Avg_Error', ascending=False)
        failure_df.to_csv(f"{output_dir}/failure_analysis/failure_modes.csv", index=False)
        
        print("\n  WORST PERFORMING MODELS (by avg error in worst 10%):")
        print(failure_df[['Model', 'Avg_Error', 'Max_Error', 'Num_Failures']].head(10).to_string())
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # 1. Error distribution for worst models
        ax1 = axes[0, 0]
        worst_5_models = failure_df.head(5)['Model'].tolist()
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(worst_5_models)))
        ax1.barh(range(len(worst_5_models)), failure_df.head(5)['Avg_Error'], 
                color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax1.set_yticks(range(len(worst_5_models)))
        ax1.set_yticklabels(worst_5_models, fontsize=10)
        ax1.set_xlabel('Average Error (g) in Worst 10% Predictions', fontsize=11, fontweight='bold')
        ax1.set_title('Top 5 Worst Models\nby Failure Severity', fontsize=13, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()
        
        # 2. Max error comparison
        ax2 = axes[0, 1]
        top_10_worst = failure_df.head(10).sort_values('Max_Error', ascending=False)
        colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(top_10_worst)))
        ax2.barh(range(len(top_10_worst)), top_10_worst['Max_Error'], 
                color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax2.set_yticks(range(len(top_10_worst)))
        ax2.set_yticklabels(top_10_worst['Model'], fontsize=10)
        ax2.set_xlabel('Maximum Single Error (g)', fontsize=11, fontweight='bold')
        ax2.set_title('Worst Case Predictions\n(Maximum Error)', fontsize=13, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()
        
        # 3. Feature patterns in failures (most important features)
        ax3 = axes[1, 0]
        # Select top 3 worst models
        top_3_worst = failure_df.head(3)
        
        feature_data = []
        for _, row in top_3_worst.iterrows():
            for feat in FEATURE_NAMES[:4]:  # Top 4 features
                feature_data.append({
                    'Model': row['Model'],
                    'Feature': feat,
                    'Value': row[f'Avg_{feat}']
                })
        
        feature_df = pd.DataFrame(feature_data)
        pivot_features = feature_df.pivot(index='Model', columns='Feature', values='Value')
        
        x = np.arange(len(pivot_features))
        width = 0.2
        
        for i, feat in enumerate(pivot_features.columns):
            ax3.bar(x + i*width, pivot_features[feat], width, 
                   label=feat, edgecolor='black', linewidth=1, alpha=0.8)
        
        ax3.set_xlabel('Worst Models', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Average Feature Value', fontsize=11, fontweight='bold')
        ax3.set_title('Feature Patterns in Failure Cases\n(Top 3 Worst Models)', 
                     fontsize=13, fontweight='bold')
        ax3.set_xticks(x + width * 1.5)
        ax3.set_xticklabels(pivot_features.index, fontsize=9, rotation=15, ha='right')
        ax3.legend(fontsize=9)
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Bias analysis (avg true vs avg pred in failures)
        ax4 = axes[1, 1]
        top_10_bias = failure_df.head(10)
        x_pos = np.arange(len(top_10_bias))
        width = 0.35
        
        bars1 = ax4.bar(x_pos - width/2, top_10_bias['Avg_True_Value'], width, 
                       label='Avg True Value', color='green', edgecolor='black', 
                       linewidth=1.5, alpha=0.7)
        bars2 = ax4.bar(x_pos + width/2, top_10_bias['Avg_Pred_Value'], width, 
                       label='Avg Predicted Value', color='red', edgecolor='black', 
                       linewidth=1.5, alpha=0.7)
        
        ax4.set_xlabel('Models', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Biomass (g)', fontsize=11, fontweight='bold')
        ax4.set_title('Bias in Failure Cases\n(True vs Predicted Values)', 
                     fontsize=13, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(top_10_bias['Model'], fontsize=8, rotation=45, ha='right')
        ax4.legend(fontsize=10)
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/failure_analysis/failure_mode_analysis.svg", format='svg', bbox_inches='tight')
        plt.close()
        
        # Generate detailed failure report
        with open(f"{output_dir}/failure_analysis/failure_report.txt", 'w') as f:
            f.write("="*80 + "\n")
            f.write(" FAILURE MODE ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            for _, row in failure_df.head(5).iterrows():
                f.write(f"\nModel: {row['Model']}\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Number of severe failures (top 10% errors): {row['Num_Failures']}\n")
                f.write(f"  Average error in failures: {row['Avg_Error']:.3f} g\n")
                f.write(f"  Maximum single error: {row['Max_Error']:.3f} g\n")
                f.write(f"  Biomass range in failures: {row['True_Value_Range']} g\n")
                f.write(f"  Average true value: {row['Avg_True_Value']:.3f} g\n")
                f.write(f"  Average predicted value: {row['Avg_Pred_Value']:.3f} g\n")
                f.write(f"  Prediction bias: {row['Avg_Pred_Value'] - row['Avg_True_Value']:.3f} g\n")
                f.write("\n  Feature characteristics in failure cases:\n")
                for feat in FEATURE_NAMES:
                    f.write(f"    {feat:20}: {row[f'Avg_{feat}']:.4f}\n")
                f.write("\n")
        
        print(f"  ✓ Failure analysis saved to: {output_dir}/failure_analysis/")
        
        return failure_df

# ==================== ORIGINAL VISUALIZATION SUITE (UNCHANGED) ====================

class VisualizationSuite:
    """All visualization generation - Enhanced for multi-model comprehensive plots"""
    
    @staticmethod
    def all_models_scatter_plots(prediction_data, model_list, output_path):
        """Generate scatter plots for ALL models in a single comprehensive figure"""
        n_models = len(model_list)
        n_cols = 4
        n_rows = int(np.ceil(n_models / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6*n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, model_name in enumerate(model_list):
            ax = axes[idx]
            
            # Get prediction data for this model
            pred_data = prediction_data.get(model_name)
            if pred_data is None:
                ax.axis('off')
                continue
            
            y_true = pred_data['y_true']
            y_pred = pred_data['y_pred']
            uncertainty = pred_data.get('y_std')
            
            # Scatter plot
            ax.scatter(y_true, y_pred, alpha=0.6, s=40, edgecolors='black', linewidth=0.5, c='steelblue')
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect', alpha=0.8)
            
            # Uncertainty bands if available
            if uncertainty is not None:
                sorted_idx = np.argsort(y_pred)
                ax.fill_between(y_pred[sorted_idx], 
                               y_pred[sorted_idx] - 1.96 * uncertainty[sorted_idx],
                               y_pred[sorted_idx] + 1.96 * uncertainty[sorted_idx],
                               alpha=0.15, color='orange', label='95% PI')
            
            # Metrics
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # Clean model name for display
            display_name = model_name.replace('Base_', '').replace('Novel_', '')
            
            # Add metrics text box
            textstr = f'$R^2$={r2:.3f}\nMAE={mae:.2f}g\nRMSE={rmse:.2f}g'
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_xlabel('Actual (g)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Predicted (g)', fontsize=10, fontweight='bold')
            ax.set_title(display_name, fontsize=11, fontweight='bold', pad=8)
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle('Scatter Plots: Predicted vs Actual - All Models', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {output_path}")
    
    @staticmethod
    def all_models_residual_plots(prediction_data, model_list, output_path):
        """Generate residual plots for ALL models in a single comprehensive figure"""
        n_models = len(model_list)
        n_cols = 4
        n_rows = int(np.ceil(n_models / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6*n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, model_name in enumerate(model_list):
            ax = axes[idx]
            
            pred_data = prediction_data.get(model_name)
            if pred_data is None:
                ax.axis('off')
                continue
            
            y_true = pred_data['y_true']
            y_pred = pred_data['y_pred']
            residuals = y_true - y_pred
            
            # Residual scatter
            ax.scatter(y_pred, residuals, alpha=0.6, s=40, edgecolors='black', linewidth=0.5, c='steelblue')
            ax.axhline(y=0, color='r', linestyle='--', lw=2, alpha=0.8)
            ax.axhline(y=np.std(residuals), color='orange', linestyle=':', lw=1.5, alpha=0.7)
            ax.axhline(y=-np.std(residuals), color='orange', linestyle=':', lw=1.5, alpha=0.7)
            
            display_name = model_name.replace('Base_', '').replace('Novel_', '')
            
            # Add std text
            textstr = f'σ={np.std(residuals):.2f}g'
            ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_xlabel('Predicted (g)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Residuals (g)', fontsize=10, fontweight='bold')
            ax.set_title(display_name, fontsize=11, fontweight='bold', pad=8)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle('Residual Plots - All Models', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {output_path}")
    
    @staticmethod
    def all_models_qq_plots(prediction_data, model_list, output_path):
        """Generate Q-Q plots for ALL models in a single comprehensive figure"""
        n_models = len(model_list)
        n_cols = 4
        n_rows = int(np.ceil(n_models / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6*n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, model_name in enumerate(model_list):
            ax = axes[idx]
            
            pred_data = prediction_data.get(model_name)
            if pred_data is None:
                ax.axis('off')
                continue
            
            y_true = pred_data['y_true']
            y_pred = pred_data['y_pred']
            residuals = y_true - y_pred
            
            # Q-Q plot
            stats.probplot(residuals, dist="norm", plot=ax)
            
            display_name = model_name.replace('Base_', '').replace('Novel_', '')
            ax.set_title(display_name, fontsize=11, fontweight='bold', pad=8)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle('Q-Q Plots (Normality Check) - All Models', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {output_path}")
    
    @staticmethod
    def all_models_bland_altman_plots(prediction_data, model_list, output_path):
        """Generate Bland-Altman plots for ALL models in a single comprehensive figure"""
        n_models = len(model_list)
        n_cols = 4
        n_rows = int(np.ceil(n_models / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6*n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, model_name in enumerate(model_list):
            ax = axes[idx]
            
            pred_data = prediction_data.get(model_name)
            if pred_data is None:
                ax.axis('off')
                continue
            
            y_true = pred_data['y_true']
            y_pred = pred_data['y_pred']
            
            mean_values = (y_true + y_pred) / 2
            differences = y_true - y_pred
            mean_diff = np.mean(differences)
            std_diff = np.std(differences)
            
            # Bland-Altman scatter
            ax.scatter(mean_values, differences, alpha=0.6, s=40, edgecolors='black', linewidth=0.5, c='steelblue')
            ax.axhline(mean_diff, color='blue', linestyle='-', lw=2, alpha=0.8)
            ax.axhline(mean_diff + 1.96*std_diff, color='red', linestyle='--', lw=1.5, alpha=0.8)
            ax.axhline(mean_diff - 1.96*std_diff, color='red', linestyle='--', lw=1.5, alpha=0.8)
            
            display_name = model_name.replace('Base_', '').replace('Novel_', '')
            
            # Add stats text
            textstr = f'μ={mean_diff:.2f}g\n±1.96σ=±{1.96*std_diff:.2f}g'
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_xlabel('Mean (g)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Difference (g)', fontsize=10, fontweight='bold')
            ax.set_title(display_name, fontsize=11, fontweight='bold', pad=8)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle('Bland-Altman Plots - All Models', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {output_path}")
    
    @staticmethod
    def all_models_residual_and_qq_combined(prediction_data, model_list, output_path):
        """Generate combined Residual+QQ plots for ALL models (2 columns per model)"""
        n_models = len(model_list)
        n_rows = n_models
        
        fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4*n_rows))
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for idx, model_name in enumerate(model_list):
            ax_res = axes[idx, 0]
            ax_qq = axes[idx, 1]
            
            pred_data = prediction_data.get(model_name)
            if pred_data is None:
                ax_res.axis('off')
                ax_qq.axis('off')
                continue
            
            y_true = pred_data['y_true']
            y_pred = pred_data['y_pred']
            residuals = y_true - y_pred
            
            display_name = model_name.replace('Base_', '').replace('Novel_', '')
            
            # Residual plot
            ax_res.scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.6, c='steelblue')
            ax_res.axhline(y=0, color='r', linestyle='--', lw=2.5, alpha=0.8)
            ax_res.axhline(y=np.std(residuals), color='orange', linestyle=':', lw=2, alpha=0.7, label='+1σ')
            ax_res.axhline(y=-np.std(residuals), color='orange', linestyle=':', lw=2, alpha=0.7, label='-1σ')
            ax_res.set_xlabel('Predicted (g)', fontsize=11, fontweight='bold')
            ax_res.set_ylabel('Residuals (g)', fontsize=11, fontweight='bold')
            ax_res.set_title(f'{display_name} - Residuals', fontsize=12, fontweight='bold')
            ax_res.legend(fontsize=9)
            ax_res.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # Q-Q plot
            stats.probplot(residuals, dist="norm", plot=ax_qq)
            ax_qq.set_title(f'{display_name} - Q-Q Plot', fontsize=12, fontweight='bold')
            ax_qq.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        fig.suptitle('Residual Analysis & Normality Check - All Models', 
                    fontsize=18, fontweight='bold', y=0.998)
        plt.tight_layout(rect=[0, 0, 1, 0.995])
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {output_path}")
    
    @staticmethod
    def calibration_plot(y_true, y_pred, y_std, model_name, output_path, n_bins=10):
        z_scores = np.abs((y_true - y_pred) / (y_std + 1e-10))
        bin_edges = np.percentile(y_std, np.linspace(0, 100, n_bins + 1))
        observed_freq, expected_freq = [], []
        
        for i in range(n_bins):
            mask = (y_std >= bin_edges[i]) & (y_std < bin_edges[i + 1])
            if np.sum(mask) > 0:
                obs = np.mean(z_scores[mask] <= 1.0)
                observed_freq.append(obs)
                expected_freq.append(0.6827)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(expected_freq, observed_freq, s=120, alpha=0.7, edgecolors='black', linewidth=2, c='steelblue')
        ax.plot([0, 1], [0, 1], 'r--', lw=2.5, label='Perfect Calibration')
        ax.set_xlabel('Expected Coverage', fontsize=13, fontweight='bold')
        ax.set_ylabel('Observed Coverage', fontsize=13, fontweight='bold')
        ax.set_title(f'{model_name}\nUncertainty Calibration', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.tight_layout()
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def critical_difference_diagram(results_df, metric='R2', output_path=None):
        pivot = results_df.pivot_table(values=metric, index='Iteration', columns='Model', aggfunc='first')
        ranks = pivot.rank(axis=1, ascending=False)
        avg_ranks = ranks.mean(axis=0).sort_values()
        
        try:
            n_models = len(avg_ranks)
            n_datasets = len(pivot)
            q_alpha = 2.850
            cd = q_alpha * np.sqrt(n_models * (n_models + 1) / (6 * n_datasets))
        except:
            cd = 1.0
        
        fig, ax = plt.subplots(figsize=(12, 7))
        y_pos = np.arange(len(avg_ranks))
        models = avg_ranks.index
        ax.barh(y_pos, avg_ranks.values, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models, fontsize=11)
        ax.set_xlabel('Average Rank (Lower is Better)', fontsize=13, fontweight='bold')
        ax.set_title(f'Critical Difference Diagram - {metric}\n(CD = {cd:.3f})', 
                     fontsize=15, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_xaxis()
        ax.axvline(x=avg_ranks.iloc[0] + cd, color='red', linestyle='--', lw=2.5, 
                   label=f'Critical Difference = {cd:.2f}')
        ax.legend(fontsize=11)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()

    @staticmethod
    def performance_comparison_chart(summary_df, output_path):
        metrics = ['R2', 'MAE', 'RMSE', 'MAPE']
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            data = summary_df.sort_values((metric, 'mean'), ascending=(metric != 'R2'))
            models = data['Model']
            means = data[(metric, 'mean')]
            stds = data[(metric, 'std')]
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
            bars = ax.barh(range(len(models)), means, xerr=stds, capsize=5, 
                           color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels(models, fontsize=10)
            ax.set_xlabel(f'{metric} {"Score" if metric == "R2" else ""}', 
                          fontsize=12, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            if metric == 'R2':
                ax.invert_xaxis()
        
        plt.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()

# ==================== MAIN EVALUATION FRAMEWORK (ENHANCED) ====================

class EvaluationFramework:
    def __init__(self):
        self.detailed_results = []
        
    def load_predictions(self):
        """Load all saved predictions from Part 1"""
        print("  Loading saved predictions...")
        
        # Load raw results - handle both comma and semicolon delimiters
        raw_results = pd.read_csv(f"{OUTPUT_DIR}/tables/raw_results.csv")
        
        # Check if it's semicolon-delimited
        if len(raw_results.columns) == 1 and ';' in str(raw_results.columns[0]):
            print("  Detected semicolon delimiter, reloading...")
            raw_results = pd.read_csv(f"{OUTPUT_DIR}/tables/raw_results.csv", sep=';')
        
        # Debug: Print column names
        print(f"  Columns in raw_results.csv: {raw_results.columns.tolist()}")
        
        # Normalize column names (strip whitespace, lowercase)
        raw_results.columns = raw_results.columns.str.strip().str.lower()
        
        # Check if 'model' column exists (after normalization)
        if 'model' not in raw_results.columns:
            print(f"  ERROR: 'Model' column not found!")
            print(f"  Available columns: {raw_results.columns.tolist()}")
            print(f"  First few rows:")
            print(raw_results.head())
            raise ValueError("'Model' column not found in raw_results.csv")
        
        # Load all predictions
        prediction_data = {}
        for seed in SEEDS:
            y_test = np.load(f"{OUTPUT_DIR}/predictions/seed_{seed}_y_test.npy")
            y_train = np.load(f"{OUTPUT_DIR}/predictions/seed_{seed}_y_train.npy")
            
            for model in raw_results['model'].unique():
                pred_file = f"{OUTPUT_DIR}/predictions/seed_{seed}_{model}_pred.npy"
                if os.path.exists(pred_file):
                    pred = np.load(pred_file)
                    
                    key = f"{model}_seed_{seed}"
                    prediction_data[key] = {
                        'y_true': y_test,
                        'y_pred': pred,
                        'y_train': y_train,
                        'model': model,
                        'seed': seed
                    }
                    
                    # Load uncertainty if available
                    std_file = f"{OUTPUT_DIR}/predictions/seed_{seed}_{model}_std.npy"
                    if os.path.exists(std_file):
                        prediction_data[key]['y_std'] = np.load(std_file)
        
        print(f"  ✓ Loaded {len(prediction_data)} prediction sets")
        return raw_results, prediction_data
    
    def compute_comprehensive_metrics(self, raw_results, prediction_data):
        """Compute all evaluation metrics"""
        print("\n  Computing comprehensive metrics...")
        
        evaluator = ComprehensiveEvaluator()
        enhanced_results = []
        
        for key, pred_data in prediction_data.items():
            metrics = evaluator.compute_all_metrics(pred_data['y_true'], pred_data['y_pred'])
            
            # Get train R2 from raw results (using lowercase column names)
            model_row = raw_results[
                (raw_results['model'] == pred_data['model']) & 
                (raw_results['seed'] == pred_data['seed'])
            ]
            
            if not model_row.empty:
                train_r2 = model_row['train_r2'].values[0]
                gen_gap, gen_status = evaluator.generalization_gap(train_r2, metrics['R2'])
            else:
                train_r2, gen_gap, gen_status = None, None, None
            
            enhanced_results.append({
                'Model': pred_data['model'],
                'Seed': pred_data['seed'],
                'Iteration': raw_results[raw_results['seed'] == pred_data['seed']]['iteration'].values[0],
                'R2': metrics['R2'],
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'MAPE': metrics['MAPE'],
                'AIC': metrics['AIC'],
                'BIC': metrics['BIC'],
                'Train_R2': train_r2,
                'Gen_Gap': gen_gap,
                'Gen_Status': gen_status
            })
        
        df = pd.DataFrame(enhanced_results)
        df.to_csv(f"{OUTPUT_DIR}/tables/complete_metrics.csv", index=False)
        print(f"  ✓ Complete metrics saved to: {OUTPUT_DIR}/tables/complete_metrics.csv")
        
        return df
    
    def generate_summary_statistics(self, df):
        """Generate comprehensive summary"""
        print("\n  Generating summary statistics...")
        
        # Only aggregate numeric columns (exclude Gen_Status which is text)
        numeric_cols = {
            'R2': ['mean', 'std', 'min', 'max', 'median'],
            'MAE': ['mean', 'std', 'min', 'max'],
            'RMSE': ['mean', 'std', 'min', 'max'],
            'MAPE': ['mean', 'std', 'min', 'max'],
            'AIC': ['mean', 'std'],
            'BIC': ['mean', 'std'],
            'Gen_Gap': ['mean', 'std']
        }
        
        summary = df.groupby('Model').agg(numeric_cols).round(4)
        
        # Add CV%
        for metric in ['R2', 'MAE', 'RMSE', 'MAPE']:
            summary[(metric, 'CV%')] = (summary[(metric, 'std')] / summary[(metric, 'mean')] * 100).round(2)
        
        summary_sorted = summary.sort_values(('R2', 'mean'), ascending=False)
        summary_sorted.to_csv(f"{OUTPUT_DIR}/tables/summary_statistics.csv")
        
        print("\n  TOP 10 MODELS BY R² SCORE:")
        top10 = summary_sorted.head(10)
        print(top10[[('R2', 'mean'), ('MAE', 'mean'), ('RMSE', 'mean'), ('MAPE', 'mean')]].to_string())
        
        return summary_sorted
    
    def statistical_tests(self, df):
        """All statistical hypothesis tests"""
        print("\n  Performing statistical tests...")
        
        print("\n  FRIEDMAN TEST:")
        for metric in ['R2', 'MAE', 'RMSE', 'MAPE']:
            pivot = df.pivot_table(values=metric, index='Iteration', columns='Model', aggfunc='first')
            stat, p = friedmanchisquare(*[pivot[col].values for col in pivot.columns])
            result = "SIGNIFICANT" if p < 0.05 else "Not Significant"
            print(f"    {metric:6} | χ²: {stat:8.3f}, p: {p:.6f} [{result}]")
            
            if p < 0.05:
                nemenyi = sp.posthoc_nemenyi_friedman(pivot)
                nemenyi.to_csv(f"{OUTPUT_DIR}/statistical_tests/nemenyi_{metric}.csv")
                
                # Heatmap
                plt.figure(figsize=(14, 12))
                sns.heatmap(nemenyi, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                           cbar_kws={'label': 'p-value'}, linewidths=0.5)
                plt.title(f'Nemenyi Post-hoc Test - {metric}\n(p < 0.05 = significant)', 
                         fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f"{OUTPUT_DIR}/statistical_tests/nemenyi_heatmap_{metric}.svg", format='svg', bbox_inches='tight')
                plt.close()
                
                # CD Diagram
                VisualizationSuite.critical_difference_diagram(
                    df, metric, f"{OUTPUT_DIR}/statistical_tests/cd_diagram_{metric}.svg")
        
        # Wilcoxon tests
        best_model = df.groupby('Model')['R2'].mean().idxmax()
        print(f"\n  WILCOXON SIGNED-RANK TEST ({best_model} vs Others):")
        
        best_scores = df[df['Model'] == best_model]['R2'].values
        wilcoxon_results = []
        
        for model in df['Model'].unique():
            if model != best_model:
                model_scores = df[df['Model'] == model]['R2'].values
                stat, p = wilcoxon(best_scores, model_scores)
                d, interpretation = ComprehensiveEvaluator.cohens_d(best_scores, model_scores)
                wilcoxon_results.append({
                    'Model': model,
                    'Statistic': stat,
                    'p-value': p,
                    'Significant': p < 0.05,
                    'Cohens_d': d,
                    'Effect_Size': interpretation
                })
                print(f"    {model:25} | p={p:.6f}, d={d:+.3f} ({interpretation})")
        
        wilcoxon_df = pd.DataFrame(wilcoxon_results).sort_values('p-value')
        wilcoxon_df.to_csv(f"{OUTPUT_DIR}/statistical_tests/wilcoxon_results.csv", index=False)
    
    def generate_visualizations(self, df, prediction_data):
        """Generate ALL visualizations for ALL models"""
        print("\n  Generating comprehensive visualizations for ALL models...")
        
        # Get all unique models
        all_models = sorted(df['Model'].unique())
        
        # Get last seed predictions for visualization
        last_seed = SEEDS[-1]
        model_predictions = {}
        for model in all_models:
            key = f"{model}_seed_{last_seed}"
            if key in prediction_data:
                model_predictions[model] = prediction_data[key]
        
        print(f"    Generating visualizations for {len(model_predictions)} models...")
        
        # 1. SCATTER PLOTS - All models in one figure
        print("    → Creating scatter plots (all models)...")
        VisualizationSuite.all_models_scatter_plots(
            model_predictions, all_models,
            f"{OUTPUT_DIR}/visualizations/scatter_plots_all_models.svg")
        
        # 2. RESIDUAL PLOTS - All models in one figure
        print("    → Creating residual plots (all models)...")
        VisualizationSuite.all_models_residual_plots(
            model_predictions, all_models,
            f"{OUTPUT_DIR}/visualizations/residual_plots_all_models.svg")
        
        # 3. Q-Q PLOTS - All models in one figure
        print("    → Creating Q-Q plots (all models)...")
        VisualizationSuite.all_models_qq_plots(
            model_predictions, all_models,
            f"{OUTPUT_DIR}/visualizations/qq_plots_all_models.svg")
        
        # 4. BLAND-ALTMAN PLOTS - All models in one figure
        print("    → Creating Bland-Altman plots (all models)...")
        VisualizationSuite.all_models_bland_altman_plots(
            model_predictions, all_models,
            f"{OUTPUT_DIR}/visualizations/bland_altman_plots_all_models.svg")
        
        # 5. COMBINED RESIDUAL + Q-Q - All models, 2 columns
        print("    → Creating combined residual+QQ plots (all models)...")
        VisualizationSuite.all_models_residual_and_qq_combined(
            model_predictions, all_models,
            f"{OUTPUT_DIR}/visualizations/residual_qq_combined_all_models.svg")
        
        # 6. BOXPLOTS - One per metric (SVG)
        print("    → Creating boxplots...")
        for metric in ['R2', 'MAE', 'RMSE', 'MAPE']:
            plt.figure(figsize=(16, 8))
            model_order = df.groupby('Model')[metric].median().sort_values(
                ascending=(metric != 'R2')).index
            
            sns.boxplot(data=df, x='Model', y=metric, order=model_order, palette="Set3")
            plt.xticks(rotation=45, ha='right')
            plt.title(f'{metric} Distribution across {len(SEEDS)} Seeds', fontsize=15, fontweight='bold')
            plt.xlabel('Model', fontsize=12, fontweight='bold')
            plt.ylabel(metric, fontsize=12, fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/visualizations/boxplot_{metric}.svg", format='svg', bbox_inches='tight')
            plt.close()
        
        # 7. PERFORMANCE COMPARISON CHART (SVG)
        print("    → Creating performance comparison chart...")
        # Only aggregate numeric columns
        numeric_cols = ['R2', 'MAE', 'RMSE', 'MAPE', 'AIC', 'BIC', 'Train_R2', 'Gen_Gap']
        summary = df.groupby('Model')[numeric_cols].agg(['mean', 'std']).reset_index()
        VisualizationSuite.performance_comparison_chart(
            summary, f"{OUTPUT_DIR}/visualizations/performance_comparison.svg")
        
        print(f"  ✓ All visualizations saved to: {OUTPUT_DIR}/visualizations/ (SVG format)")
    
    def stability_analysis(self, df):
        """Stability and generalization analysis"""
        print("\n  Analyzing stability & generalization...")
        
        stability_metrics = []
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model]
            
            stability_metrics.append({
                'Model': model,
                'R2_Mean': model_data['R2'].mean(),
                'R2_Std': model_data['R2'].std(),
                'R2_CV%': (model_data['R2'].std() / model_data['R2'].mean() * 100),
                'MAE_Mean': model_data['MAE'].mean(),
                'Gen_Gap_Mean': model_data['Gen_Gap'].mean(),
                'Gen_Gap_Std': model_data['Gen_Gap'].std()
            })
        
        stability_df = pd.DataFrame(stability_metrics).sort_values('R2_CV%')
        stability_df.to_csv(f"{OUTPUT_DIR}/stability/stability_analysis.csv", index=False)
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # CV% plot
        stability_sorted = stability_df.sort_values('R2_CV%')
        ax1.barh(range(len(stability_sorted)), stability_sorted['R2_CV%'], 
                color='steelblue', edgecolor='black', alpha=0.8)
        ax1.set_yticks(range(len(stability_sorted)))
        ax1.set_yticklabels(stability_sorted['Model'], fontsize=9)
        ax1.set_xlabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Model Stability (Lower CV% = More Stable)', fontsize=13, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Generalization gap
        gap_data = stability_df.dropna(subset=['Gen_Gap_Mean']).sort_values('Gen_Gap_Mean')
        colors = ['green' if abs(x) < 0.05 else 'orange' if x > 0.05 else 'red' 
                 for x in gap_data['Gen_Gap_Mean']]
        ax2.barh(range(len(gap_data)), gap_data['Gen_Gap_Mean'], color=colors, 
                edgecolor='black', alpha=0.8)
        ax2.set_yticks(range(len(gap_data)))
        ax2.set_yticklabels(gap_data['Model'], fontsize=9)
        ax2.set_xlabel('Generalization Gap (Train R² - Test R²)', fontsize=12, fontweight='bold')
        ax2.set_title('Overfitting Analysis\n(Green=Good, Orange=Moderate, Red=Overfitting)', 
                     fontsize=13, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
        ax2.axvline(x=0.05, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
        ax2.axvline(x=-0.05, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/stability/stability_and_generalization.svg", format='svg', bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Stability analysis saved to: {OUTPUT_DIR}/stability/")
    
    def uncertainty_evaluation(self, prediction_data):
        """Evaluate uncertainty quantification"""
        print("\n  Evaluating uncertainty quantification...")
        
        unc_eval = UncertaintyEvaluator()
        uncertainty_results = []
        
        for key, pred_data in prediction_data.items():
            if 'y_std' in pred_data and 'KAN_Uncertainty' in pred_data['model']:
                picp = unc_eval.prediction_interval_coverage(
                    pred_data['y_true'], pred_data['y_pred'], pred_data['y_std'])
                mpiw = unc_eval.mean_prediction_interval_width(pred_data['y_std'])
                cwc, _, _ = unc_eval.coverage_width_criterion(
                    pred_data['y_true'], pred_data['y_pred'], pred_data['y_std'])
                
                uncertainty_results.append({
                    'Seed': pred_data['seed'],
                    'PICP': picp,
                    'MPIW': mpiw,
                    'CWC': cwc
                })
        
        if uncertainty_results:
            unc_df = pd.DataFrame(uncertainty_results)
            unc_df.to_csv(f"{OUTPUT_DIR}/uncertainty/uncertainty_metrics.csv", index=False)
            
            print(f"  UNCERTAINTY METRICS (KAN_Uncertainty):")
            print(f"    PICP (target 0.95): {unc_df['PICP'].mean():.4f} ± {unc_df['PICP'].std():.4f}")
            print(f"    MPIW: {unc_df['MPIW'].mean():.3f} ± {unc_df['MPIW'].std():.3f}g")
            print(f"    CWC: {unc_df['CWC'].mean():.3f} ± {unc_df['CWC'].std():.3f}")
            
            # Calibration plot (use last seed)
            last_key = f"Novel_KAN_Uncertainty_seed_{SEEDS[-1]}"
            if last_key in prediction_data:
                pred_data = prediction_data[last_key]
                VisualizationSuite.calibration_plot(
                    pred_data['y_true'], pred_data['y_pred'], pred_data['y_std'],
                    "KAN_Uncertainty",
                    f"{OUTPUT_DIR}/uncertainty/calibration_plot.svg")
            
            print(f"  ✓ Uncertainty evaluation saved to: {OUTPUT_DIR}/uncertainty/")
        else:
            print("  ⚠ No uncertainty data found")
    
    def run_complete_evaluation(self):
        """Main evaluation pipeline - ENHANCED WITH NEW ANALYSES"""
        print("\n" + "="*70)
        print("  PART 2: COMPREHENSIVE EVALUATION (ENHANCED)")
        print("="*70)
        
        # Load data
        raw_results, prediction_data = self.load_predictions()
        
        # Compute metrics
        df = self.compute_comprehensive_metrics(raw_results, prediction_data)
        
        # Generate summary
        summary = self.generate_summary_statistics(df)
        
        # Statistical tests
        self.statistical_tests(df)
        
        # Visualizations
        self.generate_visualizations(df, prediction_data)
        
        # Stability analysis
        self.stability_analysis(df)
        
        # Uncertainty evaluation
        self.uncertainty_evaluation(prediction_data)
        
        # ========== NEW ANALYSES ==========
        
        # 1. Spearman Correlation Analysis
        spearman_df = SpearmanAnalyzer.compute_spearman_for_all_models(prediction_data)
        spearman_df.to_csv(f"{OUTPUT_DIR}/spearman/spearman_all_models.csv", index=False)
        spearman_summary = SpearmanAnalyzer.generate_spearman_report(spearman_df, OUTPUT_DIR)
        
        # 2. Runtime Analysis
        if os.path.exists(f"{OUTPUT_DIR}/tables/runtime_metrics.csv"):
            runtime_df = pd.read_csv(f"{OUTPUT_DIR}/tables/runtime_metrics.csv")
            RuntimeAnalyzer.analyze_runtime(runtime_df, OUTPUT_DIR)
        else:
            print("  ⚠ Runtime metrics not found. Please run Part 1 with enhanced version.")
        
        # 3. SHAP Analysis
        shap_summary = SHAPAnalyzer.aggregate_shap_values(OUTPUT_DIR, FEATURE_NAMES)
        if shap_summary:
            overall_importance = SHAPAnalyzer.visualize_shap_summary(shap_summary, OUTPUT_DIR)
            # Save overall importance
            overall_importance.to_csv(f"{OUTPUT_DIR}/shap_analysis/overall_importance.csv")
        else:
            print("  ⚠ No SHAP values found. Please run Part 1 with enhanced version.")
        
        # 4. Feature-Biomass Correlation
        correlation_summary = FeatureCorrelationAnalyzer.compute_feature_correlations(OUTPUT_DIR)
        
        # 5. Ablation Study
        ablation_df = AblationAnalyzer.analyze_feature_ablation(OUTPUT_DIR)
        
        # 6. Failure Mode Analysis
        failure_df = FailureModeAnalyzer.identify_failure_modes(prediction_data, OUTPUT_DIR)
        
        print("\n" + "="*70)
        print("  ✅ COMPREHENSIVE EVALUATION COMPLETE!")
        print(f"\n  📁 All results saved to: {OUTPUT_DIR}/")
        print("  📊 Check these directories:")
        print(f"     • {OUTPUT_DIR}/tables/ - All metric tables")
        print(f"     • {OUTPUT_DIR}/visualizations/ - Performance plots")
        print(f"     • {OUTPUT_DIR}/statistical_tests/ - Statistical analysis")
        print(f"     • {OUTPUT_DIR}/stability/ - Stability & generalization")
        print(f"     • {OUTPUT_DIR}/uncertainty/ - Uncertainty quantification")
        print(f"     • {OUTPUT_DIR}/spearman/ - Spearman correlation analysis")
        print(f"     • {OUTPUT_DIR}/runtime/ - Runtime performance analysis")
        print(f"     • {OUTPUT_DIR}/shap_analysis/ - SHAP feature importance")
        print(f"     • {OUTPUT_DIR}/feature_correlation/ - Feature-biomass correlation")
        print(f"     • {OUTPUT_DIR}/ablation/ - Feature ablation study")
        print(f"     • {OUTPUT_DIR}/failure_analysis/ - Failure mode analysis")
        print("="*70 + "\n")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  EAAI FRAMEWORK - PART 2: EVALUATION (ENHANCED)")
    print("="*70 + "\n")
    
    if not os.path.exists(f"{OUTPUT_DIR}/tables/raw_results.csv"):
        print("❌ ERROR: Part 1 must be completed first!")
        print("   Run part1_training_ENHANCED.py to train models and save predictions.")
    else:
        framework = EvaluationFramework()
        framework.run_complete_evaluation()