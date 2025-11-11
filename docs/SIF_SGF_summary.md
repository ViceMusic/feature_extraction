# SIF / SGF Peptide Stability Study Summary

## Background
Peptide drugs face rapid enzymatic degradation in the gastrointestinal tract, limiting oral delivery. Simulated Gastric Fluid (SGF, pH ~1.2 with pepsin) mimics stomach conditions, and Simulated Intestinal Fluid (SIF, pH ~6.8 with pancreatin) mimics small intestine conditions. These systems are used to evaluate peptide stability and guide optimization strategies for oral administration.

## Key Observations
- Large peptides (>12 aa) degrade rapidly in SGF/SIF, often within minutes.
- Small cyclic peptides may remain stable for hours.
- Enzymatic proteolysis, rather than pH alone, is the primary determinant of instability.
- Cyclic structures and disulfide bonds reduce protease accessibility, improving stability.

## Structural Strategies for Improving Stability
- **Cyclization** (head-to-tail, side-chain, or thioether ring formation)
- **D-amino acid substitution**
- **N-methylation**
- **Disulfide bonding**
These modifications limit conformational flexibility and block enzymatic cleavage sites.

## Machine Learning for Predicting Stability
A model using RDKit-derived molecular features and recursive feature elimination trained on 109 peptides showed:
- SGF: Accuracy ~75.1%, F1 ~84.5%
- SIF: Accuracy ~69.3%, F1 ~73.4%
Key predictive factors: lipophilicity, rigidity, molecular size.

## Dataset Construction
Human and rat peptide half-life datasets were constructed using PepLife and PepMSND databases:
- Contain peptide length, modification types, cyclic status, protease identities, and experimental details.

## Model Development
- Input features: Morgan fingerprints, Avalon fingerprints, and QED attributes.
- Classifier: XGBoost, 5-fold cross-validation.
- Data imbalance observed (long half-life samples fewer), leading to bias toward short-life predictions.

## Analysis of SIF/SGF Datasets
Four datasets (SIF1, SGF1, SIF2, SGF2) were evaluated:
- SIF datasets show higher LogP and smaller molecular weight distributions.
- SGF datasets show distinct distribution variance influenced by protease susceptibility.

## Patent Landscape
Patents reviewed include:
- Serum-stability optimization methods
- Oral delivery formulation strategies
- Peptide backbone modification chemistry

## Significance
SIF/SGF evaluation provides:
- Rapid screening of candidate drug stability
- Insight into oral bioavailability potential
- Guidance for structural optimization
- Reduced reliance on animal studies

## References
Key works cited include reviews on oral peptide delivery, gastrointestinal peptide stability, and machine learning applications in biopharmaceutical development.
