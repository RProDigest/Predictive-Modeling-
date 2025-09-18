# German Credit Risk Analysis with XGBoost and mlr3

This repository contains an R script developed by **Mubanga Nsofu (18.09.2025)** to analyze the **German Credit dataset** using the **mlr3** ecosystem.  
The script demonstrates how to train and tune an **XGBoost classifier** with Bayesian optimization, and how to evaluate performance using both **ROC** and **Precision–Recall (PR)** curves.  
All plots are styled with the **Okabe–Ito color-blind friendly palette** for professional and accessible visualization.

---

## ✨ Features

- **Data**: German Credit dataset (`tsk("german_credit")` from `mlr3`).
- **Preprocessing**: One-hot encoding pipeline with `mlr3pipelines`.
- **Model**: Gradient boosting (`classif.xgboost`) tuned with **Bayesian optimisation** (`mlr3mbo`).
- **Evaluation**:
  - AUC-ROC and confusion matrices at different thresholds.
  - Precision, Recall, and F1-scores with threshold sweep.
  - ROC curve (XGBoost vs. random baseline).
  - Precision–Recall curve with **Average Precision (AP)**.
- **Visualization**: Plots styled using the **Okabe–Ito palette** for clarity and accessibility.

---

## 📊 Example Outputs

- **ROC Curve**  
  Shows model separability with AUC score.  
  Dashed line = random classifier.  

- **Precision–Recall Curve**  
  Highlights trade-offs between recall and precision.  
  Includes Average Precision (AP) and baseline prevalence line.  

---

## 🛠️ Requirements

This project uses **R (≥4.3)** and the following packages:

```r
install.packages(c(
  "mlr3", "mlr3learners", "mlr3tuning", "mlr3mbo",
  "mlr3pipelines", "mlr3viz", "ggplot2", "pROC", "rgenoud"
))
```

---

## 🚀 Usage

Clone the repository and run the script in R:

```bash
git clone https://github.com/RProDigest/Predictive-Modeling-.git
cd Predictive-Modeling-
Rscript German_Credit_XGBoost.R
```

The script will:
1. Train an XGBoost model on the German Credit dataset.
2. Auto-tune hyperparameters with Bayesian optimization.
3. Output AUC, recall, precision, F1, and confusion matrices.
4. Generate ROC and PR plots with Okabe–Ito styling.

---

## 📖 Reference

If you use this repository, please cite:

Bischl, B., Sonabend, R., Kotthoff, L. and Lang, M. (eds.) (2024) *Applied Machine Learning Using mlr3 in R*. Boca Raton: CRC Press. Available at: <https://mlr3book.mlr-org.com> (Accessed: 18 September 2025).

---

## 👨‍💻 Author

Created by **Mubanga Nsofu**  
Date: *18 September 2025*  
Dataset: *German Credit Data*  
