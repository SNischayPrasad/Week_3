# ============================================================
# WEEK 3 TASK: UNSUPERVISED LEARNING & MODEL EVALUATION
# ============================================================
#
# Topics:
# 1. K-Means Clustering
# 2. Hierarchical Clustering
# 3. Elbow Method
# 4. Silhouette Score
# 5. PCA Dimensionality Reduction
# 6. Logistic Regression
# 7. Cross Validation
# 8. Confusion Matrix
# 9. Precision, Recall, F1 Score
# 10. ROC-AUC
# 11. Hyperparameter Tuning
#
# Dataset: Iris Dataset
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris

# Clustering
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Dimensionality Reduction
from sklearn.decomposition import PCA

# Data splitting
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV
)

# Classification
from sklearn.linear_model import LogisticRegression

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc
)

# Pipeline and preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ------------------------------------------------------------
# 2. LOAD THE DATASET
# ------------------------------------------------------------

print("=" * 70)
print("WEEK 3: UNSUPERVISED LEARNING & MODEL EVALUATION")
print("=" * 70)

print("\nLoading Iris Dataset...")

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

print("\nDataset loaded successfully!")

print("\nFeature Names:")
for feature in feature_names:
    print("-", feature)

print("\nTarget Classes:")
for i, target in enumerate(target_names):
    print(i, "=", target)

print("\nDataset Shape:")
print(X.shape)


# ------------------------------------------------------------
# 3. CREATE A DATAFRAME
# ------------------------------------------------------------

df = pd.DataFrame(X, columns=feature_names)

df["target"] = y

df["target_name"] = df["target"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


# ------------------------------------------------------------
# 4. CHECK FOR MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

print(df.isnull().sum())


# ------------------------------------------------------------
# 5. EXPLORATORY DATA VISUALIZATION
# ------------------------------------------------------------

print("\nGenerating dataset visualization...")

sns.pairplot(
    df,
    hue="target_name",
    diag_kind="hist"
)

plt.suptitle(
    "Iris Dataset Feature Relationships",
    y=1.02
)

plt.show()


# ------------------------------------------------------------
# 6. STANDARDIZE THE DATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA STANDARDIZATION")
print("=" * 70)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Data standardized successfully.")

print("\nMean after scaling:")
print(X_scaled.mean(axis=0))

print("\nStandard deviation after scaling:")
print(X_scaled.std(axis=0))


# ============================================================
# PART A: K-MEANS CLUSTERING
# ============================================================

print("\n" + "=" * 70)
print("PART A: K-MEANS CLUSTERING")
print("=" * 70)


# ------------------------------------------------------------
# 7. ELBOW METHOD
# ------------------------------------------------------------

print("\nCalculating inertia for different K values...")

inertias = []

K_values = range(1, 11)

for k in K_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertias.append(kmeans.inertia_)


# Plot Elbow Curve

plt.figure(figsize=(8, 5))

plt.plot(
    K_values,
    inertias,
    marker="o"
)

plt.title("K-Means Elbow Method")

plt.xlabel("Number of Clusters (K)")

plt.ylabel("Inertia")

plt.xticks(K_values)

plt.grid(True)

plt.show()


# ------------------------------------------------------------
# 8. SILHOUETTE SCORE FOR DIFFERENT K
# ------------------------------------------------------------

print("\nCalculating silhouette scores...")

silhouette_scores = []

for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores.append(score)

    print(
        f"K = {k}, "
        f"Silhouette Score = {score:.4f}"
    )


# Plot silhouette scores

plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    silhouette_scores,
    marker="o"
)

plt.title("Silhouette Score for K-Means")

plt.xlabel("Number of Clusters")

plt.ylabel("Silhouette Score")

plt.grid(True)

plt.show()


# ------------------------------------------------------------
# 9. TRAIN FINAL K-MEANS MODEL
# ------------------------------------------------------------

print("\nTraining final K-Means model...")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled)

print("K-Means clustering completed.")

print("\nCluster Centers:")
print(kmeans.cluster_centers_)

kmeans_silhouette = silhouette_score(
    X_scaled,
    kmeans_labels
)

print(
    "\nK-Means Silhouette Score:",
    round(kmeans_silhouette, 4)
)


# ------------------------------------------------------------
# 10. VISUALIZE K-MEANS USING PCA
# ------------------------------------------------------------

print("\nApplying PCA for visualization...")

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(X_scaled)

print(
    "Explained variance ratio:",
    pca.explained_variance_ratio_
)

print(
    "Total variance explained:",
    round(
        pca.explained_variance_ratio_.sum(),
        4
    )
)


plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_labels,
    cmap="viridis",
    s=70
)

plt.title("K-Means Clustering using PCA")

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.colorbar(
    label="Cluster"
)

plt.grid(True)

plt.show()


# ============================================================
# PART B: HIERARCHICAL CLUSTERING
# ============================================================

print("\n" + "=" * 70)
print("PART B: HIERARCHICAL CLUSTERING")
print("=" * 70)


# ------------------------------------------------------------
# 11. AGGLOMERATIVE CLUSTERING
# ------------------------------------------------------------

hierarchical = AgglomerativeClustering(
    n_clusters=3
)

hierarchical_labels = hierarchical.fit_predict(
    X_scaled
)

print(
    "Hierarchical clustering completed."
)


# Calculate silhouette score

hierarchical_silhouette = silhouette_score(
    X_scaled,
    hierarchical_labels
)

print(
    "Hierarchical Clustering Silhouette Score:",
    round(
        hierarchical_silhouette,
        4
    )
)


# ------------------------------------------------------------
# 12. VISUALIZE HIERARCHICAL CLUSTERING
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=hierarchical_labels,
    cmap="plasma",
    s=70
)

plt.title(
    "Hierarchical Clustering using PCA"
)

plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.colorbar(
    label="Cluster"
)

plt.grid(True)

plt.show()


# ------------------------------------------------------------
# 13. COMPARE CLUSTERING MODELS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLUSTERING MODEL COMPARISON")
print("=" * 70)

clustering_results = pd.DataFrame({

    "Model": [
        "K-Means",
        "Hierarchical Clustering"
    ],

    "Silhouette Score": [
        kmeans_silhouette,
        hierarchical_silhouette
    ]

})

print(
    clustering_results
)


# Plot comparison

plt.figure(figsize=(8, 5))

sns.barplot(
    data=clustering_results,
    x="Model",
    y="Silhouette Score"
)

plt.title(
    "Clustering Model Comparison"
)

plt.ylabel(
    "Silhouette Score"
)

plt.ylim(0, 1)

plt.show()


# ============================================================
# PART C: SUPERVISED MODEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("PART C: SUPERVISED MODEL EVALUATION")
print("=" * 70)


# ------------------------------------------------------------
# 14. TRAIN TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))

print("Testing samples:", len(X_test))


# ------------------------------------------------------------
# 15. CREATE LOGISTIC REGRESSION PIPELINE
# ------------------------------------------------------------

logistic_pipeline = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "model",
        LogisticRegression(
            max_iter=1000
        )
    )

])


# ------------------------------------------------------------
# 16. TRAIN MODEL
# ------------------------------------------------------------

print("\nTraining Logistic Regression...")

logistic_pipeline.fit(
    X_train,
    y_train
)

print(
    "Training completed."
)


# ------------------------------------------------------------
# 17. MAKE PREDICTIONS
# ------------------------------------------------------------

y_pred = logistic_pipeline.predict(
    X_test
)

y_probability = logistic_pipeline.predict_proba(
    X_test
)


# ------------------------------------------------------------
# 18. ACCURACY
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    "\nAccuracy:",
    round(accuracy, 4)
)


# ------------------------------------------------------------
# 19. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)


plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names
)

plt.title(
    "Confusion Matrix - Logistic Regression"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

plt.show()


# ------------------------------------------------------------
# 20. PRECISION
# ------------------------------------------------------------

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

print(
    "Precision:",
    round(precision, 4)
)


# ------------------------------------------------------------
# 21. RECALL
# ------------------------------------------------------------

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

print(
    "Recall:",
    round(recall, 4)
)


# ------------------------------------------------------------
# 22. F1 SCORE
# ------------------------------------------------------------

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print(
    "F1 Score:",
    round(f1, 4)
)


# ------------------------------------------------------------
# 23. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_names
    )
)


# ============================================================
# PART D: ROC-AUC
# ============================================================

print("\n" + "=" * 70)
print("ROC-AUC EVALUATION")
print("=" * 70)


# Iris has 3 classes, therefore we use One-vs-Rest ROC-AUC.

roc_auc = roc_auc_score(
    y_test,
    y_probability,
    multi_class="ovr",
    average="weighted"
)

print(
    "Weighted Multiclass ROC-AUC:",
    round(roc_auc, 4)
)


# ------------------------------------------------------------
# 24. PLOT ROC CURVES
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

for i in range(3):

    # Convert class labels into binary values

    y_test_binary = (
        y_test == i
    ).astype(int)

    fpr, tpr, _ = roc_curve(
        y_test_binary,
        y_probability[:, i]
    )

    roc_auc_class = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=(
            f"{target_names[i]} "
            f"(AUC = {roc_auc_class:.2f})"
        )
    )


# Random classifier line

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Multiclass ROC Curves"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# PART E: CROSS VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("CROSS-VALIDATION")
print("=" * 70)


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


cv_scores = cross_val_score(
    logistic_pipeline,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)


print("\nCross-validation scores:")

for i, score in enumerate(
    cv_scores,
    start=1
):

    print(
        f"Fold {i}: {score:.4f}"
    )


print(
    "\nMean CV Accuracy:",
    round(
        cv_scores.mean(),
        4
    )
)

print(
    "CV Standard Deviation:",
    round(
        cv_scores.std(),
        4
    )
)


# Plot CV scores

plt.figure(figsize=(8, 5))

plt.bar(
    range(1, 6),
    cv_scores
)

plt.axhline(
    cv_scores.mean(),
    linestyle="--",
    label="Mean Accuracy"
)

plt.title(
    "5-Fold Cross Validation"
)

plt.xlabel(
    "Fold"
)

plt.ylabel(
    "Accuracy"
)

plt.xticks(
    range(1, 6)
)

plt.legend()

plt.grid(
    axis="y"
)

plt.show()


# ============================================================
# PART F: HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING")
print("=" * 70)


# ------------------------------------------------------------
# 25. GRID SEARCH
# ------------------------------------------------------------

parameter_grid = {

    "model__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ],

    "model__solver": [
        "lbfgs",
        "liblinear"
    ]

}


grid_search = GridSearchCV(

    logistic_pipeline,

    parameter_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1

)


print(
    "\nRunning GridSearchCV..."
)

grid_search.fit(
    X_train,
    y_train
)

print(
    "Hyperparameter tuning completed."
)


# ------------------------------------------------------------
# 26. BEST PARAMETERS
# ------------------------------------------------------------

print(
    "\nBest Parameters:"
)

print(
    grid_search.best_params_
)


print(
    "\nBest Cross-Validation Score:"
)

print(
    round(
        grid_search.best_score_,
        4
    )
)


# ------------------------------------------------------------
# 27. EVALUATE TUNED MODEL
# ------------------------------------------------------------

best_model = grid_search.best_estimator_

tuned_predictions = best_model.predict(
    X_test
)

tuned_probabilities = best_model.predict_proba(
    X_test
)


tuned_accuracy = accuracy_score(
    y_test,
    tuned_predictions
)

tuned_precision = precision_score(
    y_test,
    tuned_predictions,
    average="weighted"
)

tuned_recall = recall_score(
    y_test,
    tuned_predictions,
    average="weighted"
)

tuned_f1 = f1_score(
    y_test,
    tuned_predictions,
    average="weighted"
)

tuned_roc_auc = roc_auc_score(
    y_test,
    tuned_probabilities,
    multi_class="ovr",
    average="weighted"
)


print("\nTuned Model Results:")

print(
    "Accuracy:",
    round(tuned_accuracy, 4)
)

print(
    "Precision:",
    round(tuned_precision, 4)
)

print(
    "Recall:",
    round(tuned_recall, 4)
)

print(
    "F1 Score:",
    round(tuned_f1, 4)
)

print(
    "ROC-AUC:",
    round(tuned_roc_auc, 4)
)


# ============================================================
# PART G: FINAL MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)


results = pd.DataFrame({

    "Metric": [

        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"

    ],

    "Logistic Regression": [

        accuracy,
        precision,
        recall,
        f1,
        roc_auc

    ],

    "Tuned Logistic Regression": [

        tuned_accuracy,
        tuned_precision,
        tuned_recall,
        tuned_f1,
        tuned_roc_auc

    ]

})


print(
    results.round(4)
)


# ------------------------------------------------------------
# 28. VISUALIZE MODEL COMPARISON
# ------------------------------------------------------------

results_plot = results.set_index(
    "Metric"
)

results_plot.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title(
    "Model Evaluation Comparison"
)

plt.ylabel(
    "Score"
)

plt.ylim(
    0,
    1.1
)

plt.xticks(
    rotation=0
)

plt.grid(
    axis="y"
)

plt.legend(
    title="Model"
)

plt.show()


# ============================================================
# PART H: SAVE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)


# Save clustering results

clustering_output = df.copy()

clustering_output[
    "KMeans_Cluster"
] = kmeans_labels

clustering_output[
    "Hierarchical_Cluster"
] = hierarchical_labels


clustering_output.to_csv(
    "clustering_results.csv",
    index=False
)


# Save classification results

classification_output = pd.DataFrame({

    "Actual": y_test,

    "Predicted": y_pred

})


classification_output.to_csv(
    "classification_results.csv",
    index=False
)


# Save model comparison

results.to_csv(
    "model_evaluation_results.csv",
    index=False
)


print(
    "Saved: clustering_results.csv"
)

print(
    "Saved: classification_results.csv"
)

print(
    "Saved: model_evaluation_results.csv"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL PROJECT SUMMARY")
print("=" * 70)


print(
    f"""
Dataset:
Iris Dataset

Number of samples:
{X.shape[0]}

Number of features:
{X.shape[1]}

K-Means Silhouette Score:
{kmeans_silhouette:.4f}

Hierarchical Clustering Silhouette Score:
{hierarchical_silhouette:.4f}

Logistic Regression Accuracy:
{accuracy:.4f}

Logistic Regression Precision:
{precision:.4f}

Logistic Regression Recall:
{recall:.4f}

Logistic Regression F1 Score:
{f1:.4f}

Logistic Regression ROC-AUC:
{roc_auc:.4f}

Tuned Model Accuracy:
{tuned_accuracy:.4f}

Tuned Model Precision:
{tuned_precision:.4f}

Tuned Model Recall:
{tuned_recall:.4f}

Tuned Model F1 Score:
{tuned_f1:.4f}

Tuned Model ROC-AUC:
{tuned_roc_auc:.4f}

Mean 5-Fold CV Accuracy:
{cv_scores.mean():.4f}

Best Hyperparameters:
{grid_search.best_params_}
"""
)

print("=" * 70)

print(
    "WEEK 3 PROJECT COMPLETED SUCCESSFULLY!"
)

print("=" * 70)