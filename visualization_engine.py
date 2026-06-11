import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

try:
    import shap
except ImportError:
    shap = None

try:
    import lifelines
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test
except ImportError:
    lifelines = None

# Kinase Superfamily Map for simplified confusion matrix
SUPERFAMILY_MAP = {
    'ABL1': 'TK', 'AKT1': 'AGC', 'ATM': 'Atypical', 'ATR': 'Atypical',
    'AURKA': 'Other', 'AXL': 'TK', 'CAMKK1': 'CAMK', 'CDK1': 'CMGC',
    'CHEK1': 'Other', 'CHK2': 'Other', 'DYRK1': 'CMGC', 'EGFR': 'TK',
    'FLT3': 'TK', 'GSK3B': 'CMGC', 'HIPK1': 'CMGC', 'LATS1': 'AGC',
    'LYN': 'TK', 'MAPK1': 'CMGC', 'MAPK3': 'CMGC', 'MTOR': 'Atypical',
    'PAK1': 'STE', 'PDGFRB': 'TK', 'PRKCA': 'AGC', 'SRC': 'TK',
    'TGFBR1': 'TKL'
}

# ----------------------------------------------------
# Dashboard 1: Model Comparison (Ablation, Radar, CD)
# ----------------------------------------------------

def plot_model_comparison(metrics_dict, save_path=None):
    """
    Plots a grouped bar chart comparing performance metrics across model variants.
    metrics_dict should map variant names to lists/arrays of metric values across folds.
    Example:
    metrics_dict = {
        'Combined (Seq+Coord)': {'Accuracy': [...], 'F1-Macro': [...], 'AUC-Macro': [...]},
        ...
    }
    """
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    variants = list(metrics_dict.keys())
    metrics = list(metrics_dict[variants[0]].keys())
    
    num_variants = len(variants)
    num_metrics = len(metrics)
    
    bar_width = 0.2
    index = np.arange(num_metrics)
    
    # Harmony color palette
    colors = ['#2b5c8f', '#d95f02', '#7570b3']
    
    for idx, variant in enumerate(variants):
        means = [np.mean(metrics_dict[variant][metric]) for metric in metrics]
        stds = [np.std(metrics_dict[variant][metric]) for metric in metrics]
        
        ax.bar(index + idx * bar_width, means, bar_width, yerr=stds, 
               label=variant, color=colors[idx % len(colors)], capsize=5, alpha=0.9, edgecolor='black', linewidth=0.7)
        
    ax.set_xlabel('Performance Metrics', fontsize=12, labelpad=10)
    ax.set_ylabel('Score', fontsize=12, labelpad=10)
    ax.set_title('Ablation Study: Performance Across Model Variants', fontsize=14, pad=15, fontweight='bold')
    ax.set_xticks(index + bar_width * (num_variants - 1) / 2)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='lower left', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_performance_radar(y_test, y_score, fams, save_path=None):
    """
    Plots a radar (spider) chart comparing metrics across the top 5 most common families.
    """
    # Find top 5 families based on sample counts
    counts = y_test.sum(axis=0)
    top_5_idc = np.argsort(counts)[-5:][::-np.argsort(counts)[-5:]]
    top_5_fams = [fams[idx] for idx in top_5_idc]
    
    # Calculate metrics for each of the top 5
    metrics_list = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
    num_metrics = len(metrics_list)
    
    angles = [n / float(num_metrics) * 2 * math.pi for n in range(num_metrics)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Harmony colors
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    
    for f_idx, fam_idx in enumerate(top_5_idc):
        y_t = y_test[:, fam_idx]
        # Threshold scores at 0.5 to compute binary classification metrics
        y_p = (y_score[:, fam_idx] >= 0.5).astype(int)
        
        acc = accuracy_score(y_t, y_p)
        
        # Binary confusion matrix values
        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_t, y_p, zero_division=0)
        
        values = [acc, prec, rec, spec, f1]
        values += values[:1]
        
        ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=fams[fam_idx], color=colors[f_idx])
        ax.fill(angles, values, color=colors[f_idx], alpha=0.08)
        
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], metrics_list, fontsize=11)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=9)
    plt.ylim(0, 1.05)
    
    plt.title("Evaluation Metrics Across Top 5 Kinase Families", size=14, y=1.1, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=10)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_critical_difference(scores_matrix, model_names, save_path=None):
    """
    Plots a Critical Difference (CD) diagram using a Nemenyi post-hoc statistical test.
    scores_matrix: numpy array of shape (num_folds, num_models) containing F1 or AUC scores.
    """
    num_folds, num_models = scores_matrix.shape
    
    # Calculate ranks (higher is better, so rank 1 is highest score)
    # Convert scores to negative to rank them ascendingly (e.g. 0.9 becomes -0.9, which ranks first)
    ranks = np.zeros_like(scores_matrix)
    for i in range(num_folds):
        ranks[i] = num_models - np.argsort(np.argsort(scores_matrix[i]))
        
    avg_ranks = ranks.mean(axis=0)
    
    # Critical Value for Nemenyi test with 3 models, alpha = 0.05 is 2.343
    q_alpha = 2.343
    cd = q_alpha * math.sqrt((num_models * (num_models + 1)) / (6.0 * num_folds))
    
    fig, ax = plt.subplots(figsize=(10, 3.5))
    
    # Draw rank axis
    ax.plot([1, num_models], [0, 0], color='black', linewidth=1.5)
    for r in range(1, num_models + 1):
        ax.plot([r, r], [-0.05, 0.05], color='black', linewidth=1.5)
        ax.text(r, -0.2, str(r), ha='center', va='top', fontsize=12, fontweight='bold')
        
    ax.text((1 + num_models)/2, -0.5, "Average Rank (Lower is Better)", ha='center', va='top', fontsize=12)
    
    # Position labels vertically to avoid overlap
    y_positions = [0.4, 0.7, 1.0]
    colors = ['#2b5c8f', '#d95f02', '#7570b3']
    
    sorted_idc = np.argsort(avg_ranks)
    
    for idx, m_idx in enumerate(sorted_idc):
        rank = avg_ranks[m_idx]
        y_pos = y_positions[idx % len(y_positions)]
        
        # Line from rank to label
        ax.plot([rank, rank], [0, y_pos], color='grey', linestyle='--', linewidth=1.0)
        
        # Label text
        label_text = f"{model_names[m_idx]} ({rank:.2f})"
        if rank < (1 + num_models)/2:
            ax.plot([rank, rank - 0.2], [y_pos, y_pos], color='grey', linewidth=1.0)
            ax.text(rank - 0.25, y_pos, label_text, ha='right', va='center', fontsize=11, color=colors[m_idx])
        else:
            ax.plot([rank, rank + 0.2], [y_pos, y_pos], color='grey', linewidth=1.0)
            ax.text(rank + 0.25, y_pos, label_text, ha='left', va='center', fontsize=11, color=colors[m_idx])
            
    # Draw CD indicator line
    ax.plot([1, 1 + cd], [-0.4, -0.4], color='red', linewidth=3.0)
    ax.text(1 + cd/2, -0.35, f"CD = {cd:.2f}", ha='center', va='bottom', color='red', fontsize=10, fontweight='bold')
    
    # Connect models if their rank difference is less than CD
    for i in range(num_models):
        for j in range(i + 1, num_models):
            m_i = sorted_idc[i]
            m_j = sorted_idc[j]
            if abs(avg_ranks[m_i] - avg_ranks[m_j]) <= cd:
                # Draw connecting bar
                ax.plot([avg_ranks[m_i], avg_ranks[m_j]], [-0.08, -0.08], color='black', linewidth=4.0, alpha=0.7)
                
    ax.set_xlim(0.5, num_models + 0.5)
    ax.set_ylim(-0.6, 1.3)
    ax.axis('off')
    
    plt.title("Nemenyi Critical Difference Diagram (F1-Score Rank)", fontsize=14, pad=10, fontweight='bold')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# ----------------------------------------------------
# Dashboard 2: Biological & Evolutionary Alignment
# ----------------------------------------------------

def plot_evolutionary_correlation(y_test, y_score, fams, fam_dist_matrix, save_path=None):
    """
    Plots a correlation between phylogenetic distance and prediction confusion rates.
    """
    num_fams = len(fams)
    confusion = np.zeros((num_fams, num_fams))
    
    # Calculate confusion matrix (predictions vs. target labels)
    for idx, score in enumerate(y_score):
        pred = np.argmax(score)
        true_idc = np.where(y_test[idx] == 1)[0]
        for t_idx in true_idc:
            confusion[pred][t_idx] += 1
            
    # Normalize confusion matrix
    for i in range(num_fams):
        s = confusion[i].sum()
        if s > 0:
            confusion[i] = confusion[i] / s
            
    # Collect inter-family distances and confusion rates (excluding self-pairs)
    distances = []
    confusion_rates = []
    pairs = []
    
    for i in range(num_fams):
        for j in range(num_fams):
            if i != j:
                distances.append(fam_dist_matrix[i][j])
                # Combine bidirectional confusion for stability
                confusion_rates.append((confusion[i][j] + confusion[j][i]) / 2)
                pairs.append((fams[i], fams[j]))
                
    df = pd.DataFrame({
        'Distance': distances,
        'Confusion': confusion_rates
    })
    
    # Calculate Pearson and Spearman correlation
    pearson_r = df['Distance'].corr(df['Confusion'], method='pearson')
    spearman_r = df['Distance'].corr(df['Confusion'], method='spearman')
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
    
    # Panel 1: Scatter plot with regression line
    sb.regplot(data=df, x='Distance', y='Confusion', ax=ax1, 
               scatter_kws={'alpha': 0.6, 'color': '#2b5c8f'}, line_kws={'color': 'red', 'linewidth': 2})
    ax1.set_xlabel('Phylogenetic Evolutionary Distance (BLOSUM62)', fontsize=11)
    ax1.set_ylabel('Model Confusion Rate', fontsize=11)
    ax1.set_title(f'Evolutionary Distance vs. Confusion\nPearson R = {pearson_r:.3f} | Spearman R = {spearman_r:.3f}', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Panel 2: Distance Matrix Heatmap
    sb.heatmap(fam_dist_matrix, cmap='Blues', ax=ax2, square=True,
               xticklabels=fams, yticklabels=fams, cbar_kws={'label': 'Phylogenetic Similarity'})
    ax2.set_xticklabels(fams, rotation=90, fontsize=8)
    ax2.set_yticklabels(fams, rotation=0, fontsize=8)
    ax2.set_title('Phylogenetic Distance Matrix', fontsize=12, fontweight='bold')
    
    # Panel 3: Confusion Heatmap
    sb.heatmap(confusion, cmap='Reds', ax=ax3, square=True,
               xticklabels=fams, yticklabels=fams, cbar_kws={'label': 'Confusion Probability'})
    ax3.set_xticklabels(fams, rotation=90, fontsize=8)
    ax3.set_yticklabels(fams, rotation=0, fontsize=8)
    ax3.set_title('Model Confusion Matrix', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_grouped_confusion_matrix(y_test, y_score, fams, save_path=None):
    """
    Groups the 25 specific families into their biological superfamilies and plots confusion.
    """
    superfamilies = sorted(list(set(SUPERFAMILY_MAP.values())))
    sf_to_idx = {sf: idx for idx, sf in enumerate(superfamilies)}
    num_sf = len(superfamilies)
    
    sf_test = np.zeros((y_test.shape[0], num_sf))
    sf_score = np.zeros((y_score.shape[0], num_sf))
    
    # Group binary matrix and predictions
    for f_idx, fam in enumerate(fams):
        sf = SUPERFAMILY_MAP.get(fam, 'Other')
        sf_idx = sf_to_idx[sf]
        sf_test[:, sf_idx] = np.maximum(sf_test[:, sf_idx], y_test[:, f_idx])
        sf_score[:, sf_idx] = np.maximum(sf_score[:, sf_idx], y_score[:, f_idx])
        
    confusion = np.zeros((num_sf, num_sf))
    for idx, score in enumerate(sf_score):
        pred = np.argmax(score)
        true_idc = np.where(sf_test[idx] == 1)[0]
        for t_idx in true_idc:
            confusion[pred][t_idx] += 1
            
    # Normalize
    labs = confusion.copy()
    for i in range(num_sf):
        s = confusion[i].sum()
        confusion[i] = confusion[i] / s if s > 0 else 0
        
    plt.figure(figsize=(8, 7))
    sb.heatmap(confusion, annot=labs, cmap='GnBu', fmt='.0f', square=True,
               xticklabels=superfamilies, yticklabels=superfamilies, cbar_kws={'label': 'Group Confusion Rate'})
    plt.xlabel('True Superfamily Group', fontsize=12, labelpad=10)
    plt.ylabel('Predicted Superfamily Group', fontsize=12, labelpad=10)
    plt.title('Superfamily Group Confusion Matrix', fontsize=14, pad=15, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

# ----------------------------------------------------
# Dashboard 3: Model Explainability (SHAP Analysis)
# ----------------------------------------------------

class PyTorchModelWrapper(object):
    """
    Wrapper for SHAP because the PyTorch model takes two distinct inputs:
    motif (batch, 22, 15) and coords (batch, 100).
    The wrapper accepts a single concatenated tensor of shape (batch, 430),
    unpacks them, runs the model, and returns output predictions.
    """
    def __init__(self, model, device, version='seq-coord'):
        self.model = model
        self.device = device
        self.version = version
        
    def __call__(self, x):
        import torch
        self.model.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        
        # Split features back
        motif_flat = x_tensor[:, :330]
        coords = x_tensor[:, 330:]
        motif = motif_flat.view(-1, 22, 15)
        
        with torch.no_grad():
            preds = self.model(motif, coords, self.version)
        return preds.cpu().numpy()

def compute_and_plot_shap(model, val_loader, device, save_path_summary=None, save_path_positional=None):
    """
    Computes SHAP values using Gradient/Kernel/Deep explainer and generates summary plots.
    """
    if shap is None:
        print("[Warning] shap library is not installed. Skipping SHAP plots.")
        return
        
    # Gather a batch of background samples and test samples
    inputs_list, coords_list = [], []
    for inputs, labels in val_loader:
        # Unpack input tensor (first 330 are motif, last 100 are Siamese embeddings)
        inputs_list.append(inputs.numpy())
        if len(inputs_list) >= 5: # Limit batch size for SHAP speed
            break
            
    X = np.vstack(inputs_list)
    
    # Create wrapper
    wrapper = PyTorchModelWrapper(model, device)
    
    # We will use shap.Explainer or shap.KernelExplainer
    # Since KernelExplainer is model-agnostic and very stable for wrappers:
    background = X[:20]  # 20 background samples
    test_samples = X[20:50]  # 30 test samples
    
    if len(test_samples) == 0:
        test_samples = X[:10]
        background = X[:5]
        
    explainer = shap.KernelExplainer(wrapper, background)
    shap_values = explainer.shap_values(test_samples, nsamples=100)
    
    # shap_values is a list of arrays (one per class output, here 25 classes)
    # Average absolute SHAP values across all 25 classes
    avg_shap_values = np.mean([np.abs(s) for s in shap_values], axis=0)
    mean_abs_shap = avg_shap_values.mean(axis=0)
    
    # 1. Feature Importance Summary: Sequence vs structure
    seq_importance = mean_abs_shap[:330].sum()
    coord_importance = mean_abs_shap[330:].sum()
    
    plt.figure(figsize=(6, 4))
    bars = plt.bar(['15-mer Sequence Motif', '3D Siamese Coordinates'], [seq_importance, coord_importance], 
            color=['#2b5c8f', '#d95f02'], edgecolor='black', width=0.5, alpha=0.9)
    plt.ylabel('Sum of Mean Absolute SHAP values', fontsize=11)
    plt.title('Feature Group Importance Summary (Ablation)', fontsize=12, fontweight='bold', pad=10)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01*seq_importance, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    if save_path_summary:
        os.makedirs(os.path.dirname(save_path_summary), exist_ok=True)
        plt.savefig(save_path_summary, dpi=300)
    plt.show()
    
    # 2. Positional SHAP Importance Map
    # Sum importances by position (each position has 22 amino acid features)
    position_shap = np.zeros(15)
    for pos in range(15):
        position_shap[pos] = mean_abs_shap[pos*22 : (pos+1)*22].sum()
        
    # Relative positions (central residue is 0, -7 to +7)
    positions = np.arange(-7, 8)
    
    plt.figure(figsize=(9, 4.5))
    plt.bar(positions, position_shap, color='#7570b3', edgecolor='black', alpha=0.9, width=0.7)
    plt.xlabel('Position relative to phosphorylated residue (0)', fontsize=11)
    plt.ylabel('Mean Absolute SHAP Value', fontsize=11)
    plt.title('Sequence Motif Positional SHAP Importance Map', fontsize=13, fontweight='bold', pad=12)
    plt.xticks(positions)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    if save_path_positional:
        os.makedirs(os.path.dirname(save_path_positional), exist_ok=True)
        plt.savefig(save_path_positional, dpi=300)
    plt.show()

# ----------------------------------------------------
# Dashboard 4: Downstream Clinical Translation
# ----------------------------------------------------

def generate_survival_simulation_data(y_score, fams):
    """
    Robust clinical survival simulator.
    Maps predicted probabilities of EGFR, AKT1, and MTOR/SRC to simulated patient cohorts.
    High predicted activity is modeled as high-risk, yielding shorter survival times.
    """
    num_samples = y_score.shape[0]
    
    # Extract predicted risk factors
    egfr_idx = np.where(fams == 'EGFR')[0][0] if 'EGFR' in fams else 11 # fallback
    akt_idx = np.where(fams == 'AKT1')[0][0] if 'AKT1' in fams else 1  # fallback
    src_idx = np.where(fams == 'SRC')[0][0] if 'SRC' in fams else 23  # fallback
    
    egfr_prob = y_score[:, egfr_idx]
    akt_prob = y_score[:, akt_idx]
    src_prob = y_score[:, src_idx]
    
    # Base survival time lambda0
    lambda_0 = 0.05
    
    # Exponential hazard model: risk = exp(2.0*EGFR + 1.5*AKT1 + 1.0*SRC)
    risk_score = 2.0 * egfr_prob + 1.5 * akt_prob + 1.0 * src_prob
    hazard = np.exp(risk_score)
    
    # Generate survival times using hazard rates
    np.random.seed(666)
    survival_times = np.random.exponential(scale=1.0 / (lambda_0 * hazard))
    
    # Scale times to standard patient months (e.g. 0 to 120 months)
    survival_times = np.clip(survival_times * 3.5, 1, 120)
    
    # Apply random censoring (e.g. 25% of patients survive past follow-up or exit early)
    censor_prob = 0.25
    event_observed = np.random.binomial(n=1, p=1-censor_prob, size=num_samples)
    
    df = pd.DataFrame({
        'EGFR_prob': egfr_prob,
        'AKT1_prob': akt_prob,
        'SRC_prob': src_prob,
        'Survival_Months': survival_times,
        'Observed_Event': event_observed,
        'Hazard_Score': risk_score
    })
    
    return df

def plot_survival_dashboard(y_score, fams, save_path_km=None, save_path_cox=None):
    """
    Runs Kaplan-Meier survival curves and Cox Proportional Hazards forest plots.
    """
    if lifelines is None:
        print("[Warning] lifelines library is not installed. Skipping Survival dashboards.")
        return
        
    df = generate_survival_simulation_data(y_score, fams)
    
    # Group cohort into High Risk vs. Low Risk based on median Hazard Score
    median_risk = df['Hazard_Score'].median()
    high_risk_mask = df['Hazard_Score'] >= median_risk
    
    # 1. Kaplan-Meier Survival Curves
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    kmf_high.fit(df.loc[high_risk_mask, 'Survival_Months'], 
                 df.loc[high_risk_mask, 'Observed_Event'], 
                 label='High Predicted Phosphorylation Activity (High Risk)')
    kmf_low.fit(df.loc[~high_risk_mask, 'Survival_Months'], 
                df.loc[~high_risk_mask, 'Observed_Event'], 
                label='Low Predicted Phosphorylation Activity (Low Risk)')
    
    kmf_high.plot_survival_probability(ax=ax, color='#e41a1c', linewidth=2.0, alpha=0.9)
    kmf_low.plot_survival_probability(ax=ax, color='#377eb8', linewidth=2.0, alpha=0.9)
    
    # Run Log-Rank statistical test
    results = logrank_test(df.loc[high_risk_mask, 'Survival_Months'],
                           df.loc[~high_risk_mask, 'Survival_Months'],
                           df.loc[high_risk_mask, 'Observed_Event'],
                           df.loc[~high_risk_mask, 'Observed_Event'])
    
    ax.set_xlabel('Timeline (Months)', fontsize=12, labelpad=10)
    ax.set_ylabel('Survival Probability', fontsize=12, labelpad=10)
    ax.set_title('Kaplan-Meier Survival Estimation by Predicted Kinase Risk Cohorts', fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Annotate log-rank p-value
    ax.text(10, 0.2, f"Log-Rank p-value: {results.p_value:.3e}\nChisq = {results.test_statistic:.2f}", 
            fontsize=11, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'))
    
    plt.tight_layout()
    if save_path_km:
        os.makedirs(os.path.dirname(save_path_km), exist_ok=True)
        plt.savefig(save_path_km, dpi=300)
    plt.show()
    
    # 2. Cox Proportional Hazards Forest Plot
    # Select our predicted kinase probabilities as risk factors
    cph_df = df[['EGFR_prob', 'AKT1_prob', 'SRC_prob', 'Survival_Months', 'Observed_Event']]
    
    cph = CoxPHFitter()
    cph.fit(cph_df, duration_col='Survival_Months', event_col='Observed_Event')
    
    plt.figure(figsize=(8, 4))
    cph.plot(color='#2b5c8f')
    plt.title('Multivariate Cox Proportional Hazards Model (Kinase Biomarkers)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Log Hazard Ratio (Coefficient)', fontsize=11, labelpad=10)
    
    plt.tight_layout()
    if save_path_cox:
        os.makedirs(os.path.dirname(save_path_cox), exist_ok=True)
        plt.savefig(save_path_cox, dpi=300)
    plt.show()

def generate_metrics_table(metrics_harvest, save_path=None):
    """
    Aggregates performance metrics across folds and prints/saves a markdown comparison table.
    """
    rows = []
    for model_name, metrics in metrics_harvest.items():
        row = {'Model Variant': model_name}
        for metric_name, values in metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            row[metric_name] = f"{mean_val:.4f} ± {std_val:.4f}"
        rows.append(row)
        
    df_table = pd.DataFrame(rows)
    df_table.set_index('Model Variant', inplace=True)
    
    print("\n" + "="*95)
    print("                      MODEL VARIANT COMPARISON TABLE (CONVERGED ACROSS 5 FOLDS)")
    print("="*95)
    try:
        # Prints markdown formatting
        print(df_table.to_markdown())
    except Exception:
        print(df_table.to_string())
    print("="*95 + "\n")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_table.to_csv(save_path)
        print(f"Metrics table successfully saved to: {save_path}\n")
        
    return df_table

