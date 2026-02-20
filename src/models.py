from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

def get_models():
    """
    Returns a dictionary of commonly used scikit-learn classification models.

    Each key is a string representing the model name, and each value is an
    untrained scikit-learn estimator instance. This function allows easy iteration
    over multiple models for training and evaluation.

    Models included:
        - "SVM Linear": Support Vector Machine with linear kernel
        - "SVM Poly"  : Support Vector Machine with polynomial kernel (degree=3)
        - "SVM RBF"   : Support Vector Machine with RBF kernel
        - "Logistic Regression": Logistic Regression classifier
        - "Decision Tree"      : Decision Tree classifier (max depth = 3)
        - "KNN"                : K-Nearest Neighbors classifier (k = 5)
        - "AdaBoost"           : AdaBoost ensemble classifier (100 estimators, learning_rate=0.5)

    Returns:
        dict: A dictionary where keys are model names (str) and values are
              untrained scikit-learn model instances.
    """
    
    return {
        "SVM Linear": SVC(kernel='linear'),
        "SVM Poly": SVC(kernel='poly', degree=3),
        "SVM RBF": SVC(kernel='rbf'),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=3),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
    }
