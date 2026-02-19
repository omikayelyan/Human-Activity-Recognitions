import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Trains and evaluates a scikit-learn classification model.

    Outputs:
        - Training Accuracy
        - Test Accuracy
        - Precision (macro)
        - Recall (macro)
        - F1-score (macro)
        - Classification Report
        - Confusion Matrix (printed + plotted)
    """

    # -------------------------
    # Train model
    # -------------------------
    model.fit(X_train, y_train)

    # -------------------------
    # Predictions
    # -------------------------
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # -------------------------
    # Metrics
    # -------------------------
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)

    precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_test, y_test_pred)

    # -------------------------
    # Print Results
    # -------------------------
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"Training Accuracy : {acc_train:.4f}")
    print(f"Test Accuracy     : {acc_test:.4f}")
    print(f"Precision (macro) : {precision:.4f}")
    print(f"Recall (macro)    : {recall:.4f}")
    print(f"F1-score (macro)  : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, zero_division=0))

    # -------------------------
    # Confusion Matrix Plot
    # -------------------------
    color_palettes = ['Blues', 'Greens', 'Reds', 'Purples', 'Oranges', 'coolwarm', 'YlGnBu']
    palette = np.random.choice(color_palettes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=palette, cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    return model
