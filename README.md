# Kinase Phosphosite 3D Detection and Structure Classifier

This repository implements a deep learning pipeline to predict kinase-specific phosphorylation sites and classify them into 25 target kinase families. It incorporates both raw sequence motifs (15-mers) and 3D structural embeddings extracted using a Siamese Neural Network.

---

## 📂 Repository Structure

*   `finalised_application.ipynb`: The main end-to-end Jupyter Notebook, split into clean, modular blocks with markdown explanations.
*   `visualization_engine.py`: A modular visualization module containing the plotting functions for the advanced dashboards.
*   `multiclassifier.py`: The PyTorch Multi-Layer Perceptron (MLP) architecture containing support for Seq-Coord, Seq-Only, and Coord-Only models.
*   `generate_dataset.py`: The preprocessing script that reads raw datasets and compiles the CSV matrices.
*   `hotfixes/`: Contains utility scripts used for modularization and layout adjustments.
*   `archive/`: Contains duplicates, legacy copies, and backup scripts.
*   `data/` (Gitignored): Holds generated data files ready for modeling.
*   `FIGS_multiclass/` (Gitignored): Directory where output analysis figures and metrics tables are saved.

---

## 🛠️ Prerequisites & Installation

To run this pipeline, install the required libraries:
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn umap-learn shap lifelines
```

---

## 🚀 How to Run the Entire Pipeline

You can run this pipeline in two ways: **interactively** (recommended for exploring visualizations) or **headlessly** via the command line.

### Step 1: Preprocess the Input Datasets
Before running the modeling code, compile the raw spreadsheets into machine-learning ready formats. Run the data preprocessing script from your terminal:
```bash
python generate_dataset.py
```
*What this does:* It parses the raw excel files (`Modified_TOP_20_FOR_STRUCTURE.xlsx` and `kinase_phosphorylation_sites.xlsx`), extracts 15-mer motifs (7 upstream and 7 downstream residues centered around the phosphosite), creates the stratified train/test folds, and generates the BLOSUM62 evolutionary distance matrices. All outputs are saved to the `data/` folder.

---

### Step 2: Choose Your Execution Method

#### Method A: Run Interactively (Jupyter Notebook) - Recommended
1.  **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```
2.  **Open the application**: Select and open **`finalised_application.ipynb`**.
3.  **Run All Cells**:
    *   Click on **Cell** in the top menu -> **Run All**, or click the **Restart & Run All** (fast-forward) button in the toolbar.
    *   *Note:* The first cell automatically installs `shap` and `lifelines` if they are missing.
4.  **Explore the Dashboards**: Scroll to the bottom of the notebook to view the interactive plots and metrics comparison tables rendered inline.

#### Method B: Run Headlessly (Terminal Script)
If you prefer running the pipeline as a background script or in a headless terminal, execute the python version:
```bash
python finalised_application.py
```
*What this does:* It trains the Siamese Network, maps UMAP coordinates, trains the 5-fold cross-validation variants, and compiles all the performance metrics.

---

### Step 3: Verify the Outputs
Once the pipeline finishes, check the output directories to verify the results:
*   **Trained Weights**: Check the `MODELS_multiclass/experiment_1/` directory for the saved PyTorch model weights (folds 0–4).
*   **Saved Visualizations**: Check the `FIGS_multiclass/` directory for all the exported 300-DPI analysis plots (CD diagrams, radar charts, SHAP positional maps, Kaplan-Meier survival curves, etc.) and the metrics comparison spreadsheet (`metrics_comparison.csv`).

---

## 📊 Advanced Visualization Suite

The pipeline automatically compiles and exports publication-grade analysis dashboards to the `FIGS_multiclass/` directory:

### 1. Model Performance & Comparison (ML Metrics)
*   **Performance Comparison Table (`metrics_comparison.csv`)**: A converged metrics summary sheet showing the mean $\pm$ standard deviation across the 5 validation folds (Accuracy, Macro/Micro Precision, Macro/Micro Recall, Macro/Micro F1, and AUC).
*   **Ablation Study Bar Plot (`ablation_study.png`)**: Compares the metrics of the Combined model vs. the Sequence-Only and Coordinate-Only baselines, quantifying the boost provided by 3D structure.
*   **Radar Chart (`radar_chart.png`)**: Spider plot mapping Accuracy, Precision, Recall, Specificity, and F1 across the top 5 most common kinase families (*AKT1*, *CDK1*, *MAPK1*, *SRC*, *AURKA*).
*   **Critical Difference (CD) Diagram (`critical_difference.png`)**: Ranks the model variants across the folds and performs Nemenyi post-hoc statistical significance testing.

### 2. Biological & Evolutionary Alignment
*   **Evolutionary Correlation Heatmaps (`evolutionary_alignment.png`)**: Scatter plot and side-by-side matrices matching the inter-family BLOSUM62 phylogenetic distance to the model's confusion rates (verifying if prediction errors align with evolutionary similarity).
*   **Grouped Superfamily Confusion Matrix (`grouped_confusion.png`)**: Heatmap grouping specific kinase families into their major superfamilies (AGC, CAMK, CMGC, TK, TKL, STE, Atypical) for high legibility.

### 3. Model Explainability (SHAP Analysis)
*   **Feature Group Summary (`shap_summary.png`)**: Summarizes the total feature contribution (Sequence vs. 3D Coordinates).
*   **Positional SHAP Map (`shap_positional.png`)**: Maps SHAP feature attribution across each of the 15 positions in the motif window, showing which flanking residues drive the classification.

### 4. Downstream Clinical Translation
*   **Kaplan-Meier Curves (`survival_km.png`)**: Simulates a patient cohort and plots patient survival curves comparing High vs. Low predicted phosphorylation activity of key biomarkers (*EGFR*, *AKT1*, *SRC*), with Log-Rank test $p$-values.
*   **Cox Proportional Hazards Forest Plot (`survival_cox.png`)**: Fits a multivariate Cox model showing Hazard Ratios (HR) and their 95% Confidence Intervals for individual predicted kinases.
