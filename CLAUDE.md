# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a peptide stability feature extraction project for SIF (Simulated Intestinal Fluid) and SGF (Simulated Gastric Fluid) datasets. The project extracts molecular features from SMILES strings and generates exploratory data analysis (EDA) visualizations for peptide stability modeling.

**Key capabilities:**
- Extract molecular features using RDKit (Morgan/Avalon fingerprints, QED properties, physicochemical descriptors)
- Process CSV files with columns: `id`, `SMILES`, `SIF_class`, `SGF_class`
- Generate comprehensive EDA visualizations
- Handle missing labels (encoded as -1) throughout the pipeline

## Environment Management

**CRITICAL:** This project uses `uv` for Python environment management.

**Always use `uv run` to execute Python scripts:**
```bash
uv run python scripts/extract_sif_sgf_features.py [args]
uv run python scripts/visualize_data.py [args]
```

**Never use:**
- `python script.py` directly
- `pip install` for dependencies

**Setup commands:**
```bash
# Sync dependencies (after pulling changes)
uv sync

# Add a new dependency
uv add package-name
```

## Project Structure

```
feature_extraction/
├── data/
│   ├── raw/              # Original CSV files (5 datasets, 1,931 samples)
│   ├── cleaned/          # Cleaned CSV files (preprocessed)
│   └── processed/        # Processed with molecular features (932 samples)
├── outputs/
│   ├── features/         # NPZ files with extracted features
│   ├── figures/          # EDA visualization plots
│   │   ├── phase1/       # Phase 1 feature quality validation
│   │   ├── phase2/       # Phase 2 within/between patent visualizations
│   │   └── phase3/       # Phase 3 model performance visualizations
│   ├── class_distribution/  # Class distribution summaries
│   └── model_results/    # Machine learning results
│       ├── cv_results/   # Cross-validation metrics (JSON)
│       ├── transfer_results/  # Transfer learning metrics (JSON)
│       ├── phase3_binary/    # Binary classification results
│       └── figures/      # Model performance visualizations
├── src/feature_extraction/
│   ├── featurizer.py     # PeptideFeaturizer class
│   ├── visualization.py  # DataVisualizer class
│   └── utils.py          # Helper functions (label conversion, molecular features)
├── notebooks/            # 🆕 Jupyter Notebooks for interactive analysis
│   ├── Phase1_数据转化.ipynb          # Interactive Phase 1 workflow
│   ├── Phase2_数据可视化.ipynb        # Interactive Phase 2 workflow
│   ├── Phase3_模型验证.ipynb          # Interactive Phase 3 workflow
│   └── README.md                      # Notebooks usage guide
├── scripts/              # Executable scripts for automation
│   ├── clean_csv_data.py              # Data cleaning
│   ├── add_molecular_features.py      # Add dimer/cyclic/disulfide features
│   ├── extract_sif_sgf_features.py    # Single file feature extraction
│   ├── extract_features.py            # Batch feature extraction
│   ├── visualize_data.py              # Basic EDA visualizations
│   ├── visualize_class.py             # Class distribution analysis
│   ├── compare_feature_distributions.py  # Compare features across datasets
│   ├── phase1_visualize.py            # Phase 1 feature quality validation
│   ├── visualize_within_patent.py     # Phase 2 within-patent analysis
│   ├── visualize_between_patents.py   # Phase 2 between-patent comparison
│   ├── generate_phase2_report.py      # Auto-generate Phase 2 report
│   ├── binary_classification.py       # Phase 3 binary classification modeling
│   ├── train_models.py                # Multi-class model training with CV
│   ├── evaluate_transfer.py           # Transfer learning evaluation
│   ├── visualize_model_results.py     # Model result visualization
│   ├── visualize_phase3_results.py    # Phase 3 comprehensive visualization
│   └── generate_phase3_report.py      # Auto-generate Phase 3 report
└── docs/
    ├── dev/                           # Development documentation
    │   ├── 项目进度.md                # Project progress tracker
    │   ├── 特征提取.md                # Feature extraction plan
    │   ├── Phase1_分子特征提取报告.md  # Phase 1 report
    │   ├── Phase2_数据可视化报告.md    # Phase 2 report
    │   └── Phase3_模型验证报告.md      # Phase 3 report
    ├── Visualization/                 # Visualization guides
    └── SIF_SGF_summary.md             # Project summary
```

## 🎯 Quick Start: Choose Your Workflow

This project offers **two complementary ways** to work with the data:

### 📓 **Jupyter Notebooks** (Recommended for Learning & Exploration)
**Use notebooks when you want to:**
- Interactively explore data and visualizations
- Learn the workflow step-by-step
- Experiment with parameters and see immediate results
- Generate presentation-ready outputs
- Debug and understand intermediate steps

**Quick Start:**
```bash
# Navigate to project root
cd /path/to/feature_extraction

# Start Jupyter Notebook
uv run jupyter notebook notebooks/

# Or use JupyterLab
uv run jupyter lab notebooks/
```

**Three notebooks available:**
1. `Phase1_数据转化.ipynb` - Data transformation and feature extraction
2. `Phase2_数据可视化.ipynb` - Exploratory data analysis and visualization
3. `Phase3_模型验证.ipynb` - Machine learning modeling and validation

**See `notebooks/README.md` for detailed usage guide.**

---

### 🖥️ **Command-Line Scripts** (Recommended for Automation)
**Use scripts when you want to:**
- Batch process multiple datasets automatically
- Run jobs in the background or on remote servers
- Integrate into automated pipelines
- Generate detailed log files
- Schedule with cron jobs

**Quick Start:**
```bash
# All scripts use 'uv run python' prefix
uv run python scripts/add_molecular_features.py --input_dir data/raw/ --output_dir data/processed/
```

---

## Common Workflows

### Complete Three-Phase Pipeline

This project follows a three-phase research pipeline as documented in `docs/dev/项目进度.md`:

**Phase 1: Data Transformation (数据转化)**
```bash
# Step 1: Add molecular features (dimer, cyclic, disulfide detection)
uv run python scripts/add_molecular_features.py \
    --input_dir data/raw/ \
    --output_dir data/processed/

# Step 2: Extract RDKit-based molecular features
uv run python scripts/extract_features.py \
    --input_dir data/processed/ \
    --output_dir outputs/features/

# Step 3: Validate feature quality
uv run python scripts/phase1_visualize.py \
    --input_dir outputs/features/ \
    --output_dir outputs/figures/phase1/
```

**Phase 2: Data Visualization (数据可视化)**
```bash
# Within-patent analysis (monomer vs dimer, structural features)
uv run python scripts/visualize_within_patent.py \
    --processed_dir data/processed/ \
    --features_dir outputs/features/ \
    --output_dir outputs/figures/phase2/within_patent/

# Between-patent comparison (PCA, t-SNE, label distributions)
uv run python scripts/visualize_between_patents.py \
    --processed_dir data/processed/ \
    --features_dir outputs/features/ \
    --output_dir outputs/figures/phase2/between_patents/

# Generate comprehensive Phase 2 report
uv run python scripts/generate_phase2_report.py \
    --figures_dir outputs/figures/phase2/ \
    --output_dir docs/dev/
```

**Phase 3: Model Validation (模型验证)**
```bash
# Binary classification with cross-validation and transfer learning
uv run python scripts/binary_classification.py \
    --processed_dir data/processed/ \
    --features_dir outputs/features/ \
    --output_dir outputs/model_results/phase3_binary/ \
    --n_folds 5 \
    --use_gpu

# Visualize Phase 3 results (confusion matrices, feature importance, heatmaps)
uv run python scripts/visualize_phase3_results.py \
    --results_dir outputs/model_results/phase3_binary/ \
    --output_dir outputs/figures/phase3/

# Generate comprehensive Phase 3 report
uv run python scripts/generate_phase3_report.py \
    --results_dir outputs/model_results/phase3_binary/ \
    --output_dir docs/dev/
```

### Legacy Multi-Class Pipeline

**For original multi-class classification workflow:**

```bash
# Step 1: Clean raw CSV data
uv run python scripts/clean_csv_data.py \
    --input data/raw/dataset.csv \
    --output data/cleaned/dataset_cleaned.csv

# Step 2: Extract features
uv run python scripts/extract_features.py \
    --input_dir data/cleaned/ \
    --output_dir outputs/features/

# Step 3: Visualize data distributions
uv run python scripts/visualize_data.py \
    --input_dir outputs/features/ \
    --output_dir outputs/figures/

# Step 4: Train models with cross-validation
uv run python scripts/train_models.py \
    --input_dir outputs/features/ \
    --output_dir outputs/model_results/cv_results/

# Step 5: Evaluate transfer learning (if multiple datasets)
uv run python scripts/evaluate_transfer.py \
    --dataset1 outputs/features/dataset1.npz \
    --dataset2 outputs/features/dataset2.npz \
    --output_dir outputs/model_results/transfer_results/

# Step 6: Visualize model results
uv run python scripts/visualize_model_results.py \
    --cv_dir outputs/model_results/cv_results/ \
    --transfer_dir outputs/model_results/transfer_results/ \
    --output_dir outputs/model_results/figures/
```

### 1. Feature Extraction Pipeline

**Extract features from cleaned CSV files:**
```bash
uv run python scripts/extract_sif_sgf_features.py \
    --input data/cleaned/sif_sgf_second_cleaned.csv \
    --output outputs/features/sif_sgf_second.npz
```

**Batch process all CSVs in a directory:**
```bash
uv run python scripts/extract_features.py \
    --input_dir data/cleaned/ \
    --output_dir outputs/features/
```

### 2. Data Visualization

**Generate EDA plots from NPZ files:**
```bash
uv run python scripts/visualize_data.py \
    --input_dir outputs/features/ \
    --output_dir outputs/figures/ \
    --dpi 300 \
    --format png
```

**Visualize class distribution:**
```bash
uv run python scripts/visualize_class.py \
    --input outputs/features/sif_sgf_second.npz \
    --output_dir outputs/class_distribution/
```

**Compare feature distributions:**
```bash
uv run python scripts/compare_feature_distributions.py \
    --input_dir outputs/features/ \
    --output_dir outputs/figures/
```

### 3. Data Cleaning

**Clean raw CSV files:**
```bash
uv run python scripts/clean_csv_data.py \
    --input data/raw/sif_sgf_second.csv \
    --output data/cleaned/sif_sgf_second_cleaned.csv
```

### 4. Model Training and Evaluation

**Train models with 5-fold cross-validation:**
```bash
uv run python scripts/train_models.py \
    --input_dir outputs/features/ \
    --output_dir outputs/model_results/cv_results/ \
    --n_folds 5
```

This trains three models (Logistic Regression, Random Forest, XGBoost) for both SIF and SGF targets on each dataset using stratified cross-validation with balanced class weights.

**Evaluate transfer learning between datasets:**
```bash
uv run python scripts/evaluate_transfer.py \
    --dataset1 outputs/features/US9624268_cleaned.npz \
    --dataset2 outputs/features/sif_sgf_second_cleaned.npz \
    --output_dir outputs/model_results/transfer_results/
```

This performs bidirectional transfer learning (train on one dataset, test on another) and handles different class definitions through class mapping.

**Visualize model results:**
```bash
uv run python scripts/visualize_model_results.py \
    --cv_dir outputs/model_results/cv_results/ \
    --transfer_dir outputs/model_results/transfer_results/ \
    --output_dir outputs/model_results/figures/
```

Generates performance comparison charts, transfer learning heatmaps, feature importance plots, and confusion matrices.

## Core Components

### Utils Module (`src/feature_extraction/utils.py`)

**Purpose:** Provides label conversion and molecular feature detection utilities.

**Key functions:**

**1. Label Conversion:**
- `convert_label_to_minutes(label)`: Converts diverse label formats to minutes
  - Star system (`*****` to `*`) → 360, 180, 60, 30, 10 minutes
  - Numeric classes (1-4) → 540, 180, 60, 30 minutes
  - Direct minute values → unchanged
  - Missing/invalid → -1

**2. Molecular Feature Detection:**
- `detect_dimer(smiles)`: Detects dimeric peptides
  - Checks for PEG linkers, high molecular weight, long SMILES
  - Returns: `True` if dimer, `False` if monomer
- `detect_cyclic(smiles)`: Detects cyclic structures
  - Checks for ring structures using RDKit
  - Returns: `True` if cyclic, `False` otherwise
- `detect_disulfide(smiles)`: Detects disulfide bonds
  - Searches for "CSSC" pattern in SMILES
  - Returns: `True` if contains disulfide, `False` otherwise
- `extract_molecular_features(smiles)`: Combined feature extraction
  - Returns: `{"is_dimer": bool, "is_cyclic": bool, "has_disulfide": bool}`

**3. Data Utilities:**
- `validate_csv_columns(df, required_cols)`: Validates CSV column structure
- `get_csv_files(directory)`: Gets all CSV files from a directory
- `load_csv_safely(csv_path)`: Safely loads CSV with error handling
- `build_output_path(input_path, output_dir, suffix)`: Constructs output paths
- `save_features_to_npz(...)`: Saves features to NPZ format
- `format_batch_summary(stats)`: Formats batch processing summary

**Usage:**
```python
from src.feature_extraction.utils import (
    convert_label_to_minutes,
    extract_molecular_features
)

# Convert labels
minutes = convert_label_to_minutes("***")  # Returns 60
minutes = convert_label_to_minutes(2)      # Returns 180

# Extract molecular features
features = extract_molecular_features("CC(C)C[C@H](NC(=O)...")
print(features)  # {"is_dimer": False, "is_cyclic": True, "has_disulfide": False}
```

### PeptideFeaturizer (`src/feature_extraction/featurizer.py`)

**Purpose:** Extract molecular features from SMILES strings.

**Key features extracted:**
1. **QED Properties (8 features):** MW, ALOGP, HBA, HBD, PSA, ROTB, AROM, ALERTS
2. **Physicochemical Descriptors (11 features):** Molecular weight, LogP, HBA, HBD, TPSA, rotatable bonds, ring count, Fsp3, heavy atom count, total atoms, rigidity proxy
3. **Gasteiger Charge Statistics (5 features):** Mean, max, min, std, sum
4. **Morgan Fingerprint (1024 bits by default)**
5. **Avalon Fingerprint (512 bits, optional):** Depends on RDKit build

**Usage:**
```python
from src.feature_extraction.featurizer import PeptideFeaturizer

featurizer = PeptideFeaturizer(
    morgan_bits=1024,
    avalon_bits=512,
    use_avalon=True
)

features, success = featurizer.featurize("CC(C)C[C@H](NC(=O)...")
feature_names = featurizer.get_feature_names()
```

**Important notes:**
- Returns `(None, False)` for invalid SMILES
- All NaN/Inf values are safely converted to 0.0
- Feature order: QED → Physchem → Gasteiger → Morgan → Avalon

### DataVisualizer (`src/feature_extraction/visualization.py`)

**Purpose:** Generate EDA visualizations from NPZ feature files.

**Key visualizations:**
1. **Feature distributions:** Histograms with KDE for MW, LogP, HBA, HBD
2. **Label distribution:** Bar charts for SIF/SGF class counts (excluding -1)
3. **Joint distribution:** Heatmap showing SIF vs SGF class co-occurrence (including -1 as "Missing")
4. **Grouped boxplots:** MW and LogP vs stability classes
5. **Correlation heatmap:** Pearson correlation for features
6. **Scatter matrix:** Pairwise feature relationships

**Missing label handling:**
- Labels with value `-1` indicate missing/unavailable data
- Bar charts filter out -1 and show missing count in title
- Heatmaps include -1 as "Missing" category

**Usage:**
```python
from src.feature_extraction.visualization import DataVisualizer

visualizer = DataVisualizer(
    X=features,
    y_sif=sif_labels,
    y_sgf=sgf_labels,
    feature_names=feature_names,
    dataset_name="SIF_SGF_Dataset",
    dpi=300
)

visualizer.plot_feature_distributions(output_dir, format="png")
visualizer.plot_label_distribution(output_dir, format="png")
summary = visualizer.generate_summary_statistics()
```

## Data Format

### Input CSV Format

**Raw CSV (data/raw/):**
```csv
id,SMILES,SIF_class,SGF_class
peptide_001,CC(C)C[C@H](NC...)...,***,**
peptide_002,C[C@H](NC(=O)...)...,2,3
```

**Processed CSV (data/processed/):**
```csv
id,SMILES,SIF_class,SGF_class,SIF_minutes,SGF_minutes,is_dimer,is_cyclic,has_disulfide
peptide_001,CC(C)C[C@H](NC...)...,***,**,60,30,0,1,0
peptide_002,C[C@H](NC(=O)...)...,2,3,180,60,1,1,1
```

**Column requirements:**

**Raw CSV:**
- `id`: Unique identifier (string/int)
- `SMILES`: Valid SMILES string
- `SIF_class`: Star/numeric/minute label (or empty for missing)
- `SGF_class`: Star/numeric/minute label (or empty for missing)

**Processed CSV (additional columns):**
- `SIF_minutes`: Half-life in minutes (or -1 for missing)
- `SGF_minutes`: Half-life in minutes (or -1 for missing)
- `is_dimer`: 1 if dimer, 0 if monomer
- `is_cyclic`: 1 if cyclic, 0 if linear
- `has_disulfide`: 1 if contains disulfide bond, 0 otherwise

### Output NPZ Format

**Saved arrays:**
- `X`: Feature matrix, shape (n_samples, n_features), dtype float32
- `y_sif`: SIF class labels, shape (n_samples,), dtype int
- `y_sgf`: SGF class labels, shape (n_samples,), dtype int
- `ids`: Sample IDs, shape (n_samples,), dtype object
- `feature_names`: Feature names, shape (n_features,), dtype object
- `mask_valid`: Boolean mask of valid samples, shape (n_rows,), dtype bool
- `metadata`: Dictionary with extraction metadata, dtype object

**Loading NPZ files:**
```python
import numpy as np

npz = np.load("outputs/features/dataset.npz", allow_pickle=True)
X = npz['X']
y_sif = npz['y_sif']
y_sgf = npz['y_sgf']
feature_names = npz['feature_names']
ids = npz['ids']
```

### Model Results Format

**Cross-validation results (JSON):**
```json
{
  "dataset_name": "sif_sgf_second",
  "target": "SIF",
  "model": "RandomForest",
  "n_folds": 5,
  "metrics": {
    "accuracy": [0.85, 0.87, ...],
    "precision": [0.83, 0.86, ...],
    "recall": [0.82, 0.85, ...],
    "f1": [0.82, 0.85, ...],
    "auc": [0.90, 0.92, ...]
  },
  "mean_metrics": {...},
  "std_metrics": {...}
}
```

**Transfer learning results (JSON):**
```json
{
  "train_dataset": "US9624268",
  "test_dataset": "sif_sgf_second",
  "target": "SIF",
  "model": "XGBoost",
  "class_mapping": {"0": 0, "1": 1, ...},
  "metrics": {
    "accuracy": 0.78,
    "precision": 0.76,
    ...
  },
  "confusion_matrix": [[...], [...]]
}
```

## Machine Learning Pipeline

### Models

Three classification models are trained for both SIF and SGF targets:

1. **Logistic Regression:** Baseline linear model with L2 regularization
2. **Random Forest:** Ensemble model with 100 trees, good for feature importance
3. **XGBoost:** Gradient boosting model, typically best performance

**All models use:**
- `class_weight='balanced'` to handle class imbalance
- Stratified K-fold cross-validation (default 5 folds)
- Standard evaluation metrics: accuracy, precision, recall, F1, AUC

### Transfer Learning

The `evaluate_transfer.py` script handles:
- Training on one dataset, testing on another (bidirectional)
- Class mapping for datasets with different class definitions (e.g., 5-class vs 4-class)
- Automatic handling of incompatible class spaces

**Class mapping example:**
- Dataset 1 has classes: [0, 1, 2, 3, 4]
- Dataset 2 has classes: [0, 1, 2, 3]
- Mapping: {0: 0, 1: 1, 2: 2, 3: 3, 4: 3} (collapse 4 into 3)

## Phase-Specific Implementation Details

### Phase 1: Molecular Feature Detection

**Dimer Detection Logic:**
The `detect_dimer()` function uses multiple heuristics:
1. **PEG Linker Detection**: Searches for common PEG patterns (e.g., "OCCOCCOCCO")
2. **Molecular Weight**: Threshold > 2000 Da suggests dimer
3. **SMILES Length**: Very long SMILES (> 300 chars) suggests dimer
4. **Multi-criteria Scoring**: Combines all signals for robust detection

**Cyclic Structure Detection:**
Uses RDKit's ring detection:
```python
mol = Chem.MolFromSmiles(smiles)
num_rings = mol.GetRingInfo().NumRings()
is_cyclic = (num_rings > 0)
```

**Disulfide Bond Detection:**
Simple pattern matching for "CSSC" motif in SMILES string.

### Phase 2: Visualization Strategies

**Within-Patent Analysis:**
- Compares monomer vs dimer stability within each dataset
- Uses Mann-Whitney U test for statistical significance
- Generates 3 plots per dataset: SIF distribution, SGF distribution, structural features

**Between-Patent Analysis:**
- Dimensionality reduction: PCA and t-SNE for 2D visualization
- Point size encodes SIF half-life, color encodes dataset origin
- Violin plots and boxplots compare label distributions across datasets
- Kruskal-Wallis test assesses cross-dataset differences

### Phase 3: Binary Classification

**Label Binarization:**
- Computes median SIF/SGF half-life for each dataset independently
- Samples ≥ median → "stable" (1), < median → "unstable" (0)
- Ensures balanced classes within each dataset

**Model Configuration:**
- **Logistic Regression**: `max_iter=1000, class_weight='balanced'`
- **Random Forest**: `n_estimators=100, class_weight='balanced', n_jobs=-1`
- **XGBoost**: `device='cuda:0', tree_method='hist', max_depth=6, learning_rate=0.1`

**GPU Acceleration:**
XGBoost uses GPU if available. Note: `gpu_id` parameter deprecated in XGBoost 3.1+, use `device='cuda:0'` instead.

**Cross-Validation:**
- Stratified K-fold (default 5 folds)
- Automatically adjusts `n_folds` if sample size < 10 (e.g., US20140294902A1 has only 5 samples)

**Transfer Learning:**
- Trains on Dataset A, tests on Dataset B (bidirectional)
- Matches class distributions between datasets
- Saves confusion matrices and transfer metrics

## Key Implementation Details

### Missing Label Handling

**Throughout the codebase:**
- Missing labels are encoded as `-1`
- `-1` is a valid value preserved during feature extraction
- Visualization scripts handle `-1` consistently:
  - Bar charts: Filter out -1, show missing count
  - Heatmaps: Include -1 as "Missing" category
  - Statistics: Report missing counts separately

### Feature Naming Convention

**Prefixes indicate feature type:**
- `QED_*`: QED properties (e.g., `QED_MW`, `QED_ALOGP`)
- `PC_*`: Physicochemical descriptors (e.g., `PC_MolWt`, `PC_LogP`)
- `GC_*`: Gasteiger charge statistics (e.g., `GC_Mean`, `GC_Max`)
- `Morgan_*`: Morgan fingerprint bits (e.g., `Morgan_0`, `Morgan_1`)
- `Avalon_*`: Avalon fingerprint bits (e.g., `Avalon_0`, `Avalon_1`)

### Error Handling

**All feature extraction functions:**
- Use try-except blocks with fallback to zero vectors
- Log warnings for individual failures (debug level)
- Never raise exceptions that stop batch processing
- Return success flags: `(features, True)` or `(None, False)`

## Research Context

This project supports peptide stability modeling in simulated gastrointestinal fluids. Key research focuses from `docs/Visualization/README.md`:

**Important physicochemical properties:**
- **Molecular Weight (MW):** Size/permeability indicator
- **LogP:** Lipophilicity/hydrophobicity
- **HBA/HBD:** Hydrogen bonding capacity
- **Rigidity proxy:** Rings / (1 + rotatable bonds)

**Performance metrics for classification:**
- Accuracy, Precision, Recall, F1-score, AUC
- Threshold variants: SIF1/SGF1 (6h), SIF2/SGF2 (2h)

**Feature importance insights:**
- Lipophilicity (LogP-related features) are highly predictive
- Molecular size and structural rigidity are key determinants
- Morgan/Avalon fingerprints capture structural patterns

## Development Notes

### When adding new features:

1. Add extraction logic to `PeptideFeaturizer.featurize()`
2. Update `PeptideFeaturizer.get_feature_names()` to include new names
3. Update `PeptideFeaturizer.n_features` property
4. Use consistent naming convention with prefixes
5. Ensure error handling returns default values (not None)

### When modifying visualizations:

1. Check if changes affect missing label handling (-1 values)
2. Update both bar charts (filter -1) and heatmaps (include -1)
3. Maintain consistent labeling: "Missing" for -1 in visualizations
4. Test with datasets containing various missing patterns

### When debugging:

**Check log files in project root:**
- `feature_extraction.log`: Feature extraction details
- `visualization.log`: Visualization generation details
- `visualize_class.log`: Class distribution analysis
- `train_models.log`: Model training and cross-validation
- `evaluate_transfer.log`: Transfer learning evaluation
- `visualize_model_results.log`: Model result visualization
- `add_molecular_features.log`: Phase 1 molecular feature detection
- `visualize_within_patent.log`: Phase 2 within-patent analysis
- `visualize_between_patents.log`: Phase 2 between-patent comparison
- `binary_classification.log`: Phase 3 binary classification
- `visualize_phase3_results.log`: Phase 3 result visualization

**Common issues:**
- Avalon not available: Check RDKit build, set `use_avalon=False`
- Invalid SMILES: Check `mask_valid` array in NPZ output
- Missing features: Verify feature names match between extraction and visualization
- Class imbalance warnings: Models use `class_weight='balanced'` by default
- Transfer learning failures: Check class mapping compatibility between datasets
- XGBoost GPU errors: Ensure CUDA is available, use `device='cuda:0'` not `gpu_id`
- ID mismatch errors: NPZ IDs are strings, CSV IDs may be int64 - convert to string for matching
- Insufficient samples for CV: Script auto-adjusts `n_folds` to `min(n_samples, n_folds)`

## Project Progress and Documentation

**Current Status:** Phase 1-3 completed (as of 2025-11-14)

**Key Metrics:**
- Original samples: 1,931 across 5 datasets
- Valid samples after processing: 932 (48.3%)
- Largest dataset: sif_sgf_second (558 samples)
- Smallest dataset: US20140294902A1 (5 samples)

**Phase Reports:**
- `docs/dev/Phase1_分子特征提取报告.md`: Feature extraction and label conversion
- `docs/dev/Phase2_数据可视化报告.md`: Within/between patent visualization analysis
- `docs/dev/Phase3_模型验证报告.md`: Binary classification and transfer learning results
- `docs/dev/项目进度.md`: Comprehensive project progress tracker

**Key Findings:**
1. **Dimer Stability**: Dimers show significantly higher stability (p < 0.05, Mann-Whitney U)
2. **Dataset Heterogeneity**: Significant differences in label distributions across datasets (p < 0.001, Kruskal-Wallis)
3. **Model Performance**: F1 scores range from 0.65-0.89 for within-dataset CV
4. **Transfer Learning**: Cross-dataset performance drops notably due to distribution shift
5. **Feature Importance**: LogP and molecular weight are most predictive features
