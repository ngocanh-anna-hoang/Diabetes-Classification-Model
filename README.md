# Diabetes Classification Modelling for Decision Support

## Overview

This project develops a machine learning workflow to analyse diabetes and prediabetes risk using health survey data. Using **Orange Data Mining** and Python, it explores the dataset, performs preprocessing, builds and evaluates decision tree and random forest classifiers, and tunes decision tree depth to investigate the trade-off between detecting diabetes cases and generalising to unseen data.

The project focuses on a key challenge in health classification: **class imbalance**. Since most respondents are classified as not having diabetes, a model may achieve seemingly strong overall performance while missing a substantial proportion of people with diabetes or prediabetes.

## Objectives

- Explore demographic, lifestyle, and health-related factors associated with diabetes classification.
- Clean inconsistent categorical values and handle missing data.
- Build and compare decision tree and random forest classifiers.
- Tune decision tree depth using stratified cross-validation.
- Evaluate minority-class recall to understand missed diabetes/prediabetes cases.
- Investigate overfitting and identify important predictive attributes.
- Recommend improvements for more reliable decision-support modelling.

## Tools & Techniques

- **Orange Data Mining** — data exploration, preprocessing, sampling, classification, and model evaluation.
- **Python** — hyperparameter tuning and class-specific recall evaluation.
- **Data preprocessing** — feature-type correction, categorical value standardisation, missing-value imputation, and identifier removal.
- **Machine learning** — decision tree and random forest classification.
- **Model evaluation** — precision, recall, F1-score, AUC, and train–test comparison.
- **Hyperparameter optimisation** — GridSearchCV with five-fold stratified cross-validation.
- **Imbalanced classification analysis** — minority-class recall and overfitting assessment.

## Dataset

The dataset contains **7,000 health survey records and 18 features**, excluding the `Index` identifier.

The target variable is `Diabetes`:

- `0` — No diabetes
- `1` — Prediabetes or diabetes

Predictor variables cover health conditions, lifestyle behaviours, demographic characteristics, and physical and mental health indicators. Examples include `High_Chol`, `BMI`, `Physical_Activity`, `Heart_Disease_or_Attack`, `Age_Group`, and `Education_Level`.

The target distribution is imbalanced: approximately **86% No and 14% Yes**.

## Data Preprocessing

The workflow included the following steps:

- Corrected the data types of `Index`, `Age_Group`, `Education_Level`, and `Chol_Checked`.
- Standardised inconsistent categorical values and converted binary variables into meaningful labels where possible.
- Identified 284 missing values across `Nbr_Days_Poor_Mental` and `Has_Healthcare`.
- Used a Simple Tree model-based imputer to estimate missing values.
- Removed `Index` because it uniquely identifies records but has no meaningful predictive value.
- Used reproducible, stratified sampling to split the data into 4,900 training records (70%) and 2,100 test records (30%).

`Age_Group` and `Education_Level` retained their original numeric codes because recoding them into category labels could not be completed in Orange. Their interpretation therefore requires caution.

## Model Development

Two baseline classifiers were developed using the training data:

- **Decision Tree:** maximum depth of 5, with constraints on leaf size and splitting.
- **Random Forest:** 10 trees, with depth and subset-size settings aligned with the baseline tree for comparison.

The models were evaluated using Orange's Test and Score widget. Because the target is imbalanced, the project also used Python's `GridSearchCV` and five-fold stratified cross-validation to tune decision tree depth specifically for recall of the `Yes` class.

## Model Results

### Baseline comparison

| Metric | Decision Tree | Random Forest |
|---|---:|---:|
| F1-score | 0.820 | 0.802 |
| Precision | 0.814 | 0.787 |
| Recall | 0.860 | 0.861 |

*These are the reported aggregate metrics averaged across both target classes, not the metrics for the diabetes/prediabetes class alone.*

The baseline decision tree achieved a slightly higher aggregate F1-score than the random forest. However, these results alone do not establish which model is better at detecting diabetes/prediabetes cases, because the majority `No` class strongly influences the aggregate metrics.

### Decision tree depth tuning

Grid search evaluated maximum depths from 5 to 33, using recall for the `Yes` class as the selection metric.

- **Depth 5:** cross-validated `Yes` recall of approximately 0.093.
- **Depth 20:** highest reported cross-validated `Yes` recall of approximately 0.304.
- Greater depths produced little further improvement and showed signs of overfitting.

The depth-20 tree performed much more strongly on training data than on the held-out test data. Its AUC fell from approximately 0.934 on training data to 0.519 on test data, indicating poor generalisation.

On the held-out test set, depth 5 achieved `Yes` recall of approximately 0.105, while depth 20 achieved approximately 0.213. Although depth 20 detected more positive cases, its overfitting makes it unreliable for deployment without further improvement.

## Key Findings

### 1. Class imbalance affects model evaluation

With around 86% of respondents in the `No` class, aggregate accuracy, precision, and recall can appear strong even when the model misses many diabetes/prediabetes cases. Class-specific recall is essential for understanding this limitation.

### 2. Decision tree depth creates a trade-off

A shallow tree generalised more consistently between training and test data but detected very few `Yes` cases. A deeper tree improved positive-class recall but overfit the training data. Increasing model complexity alone did not provide a reliable solution.

### 3. Important predictive attributes

Both decision trees split first on `High_Chol`, followed by `BMI`. Other recurring attributes included `Age_Group`, `Walk_Difficult`, and `Education_Level`.

These are predictive patterns within this dataset, not proof of causation or a clinical diagnosis.

## Recommendations

1. Investigate class-imbalance strategies such as class weighting, oversampling, or undersampling before further model tuning.
2. Evaluate precision, recall, and F1-score specifically for the `Yes` class rather than relying only on aggregate metrics.
3. Use stratified cross-validation and a separate held-out test set to assess generalisation.
4. Monitor the gap between training and test performance to detect overfitting.
5. Compare alternative models and tuning strategies after addressing class imbalance.
6. Treat model outputs as decision-support signals, not medical diagnoses. Any real-world use would require appropriate clinical validation and professional oversight.

## Conclusion

This project demonstrates how data preprocessing, model comparison, and hyperparameter tuning can support diabetes classification analysis. The results highlight that aggregate performance can conceal weak detection of the minority class and that deeper decision trees may overfit when trying to improve positive-class recall.

The current models are **not yet sufficiently reliable for standalone screening decisions**. Addressing class imbalance and improving minority-class detection and generalisation should be priorities for future development.

## References

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357.

Galar, M., Fernández, A., Barrenechea, E., Bustince, H., & Herrera, F. (2011). A review on ensembles for the class imbalance problem: Bagging-, boosting-, and hybrid-based approaches. *IEEE Transactions on Systems, Man, and Cybernetics, Part C, 42*(4), 463–484. https://doi.org/10.1109/TSMCC.2011.2161285

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
