import pandas as pd
from sklearn.preprocessing import StandardScaler

def augmentation_data(X_train: pd.DataFrame, y_train: pd.Series):
    """
        Fits a StandardScaler on the training data and creates a modified
        version of the dataset by subtracting (mean / 100) from each feature.

        The function then vertically concatenates:
            - the original training features
            - the modified feature set

        To maintain alignment, the training labels are also duplicated
        and concatenated accordingly.

        Parameters:
            X_train (DataFrame): Original training features
            y_train (Series): Original training labels

        Returns:
            X_scaled (DataFrame): Combined dataset containing original and
                                  transformed feature rows.
            y_scaled (Series): Corresponding duplicated labels.

    """

    scaler = StandardScaler()

    X_scaled = scaler.fit(X_train)

    X_scaled = X_train.values - X_scaled.mean_/100

    X_scaled = pd.DataFrame(X_scaled, columns=X_train.columns)

    X_augmented = pd.concat([X_train, X_scaled], axis=0, ignore_index=True)

    y_augmented = pd.concat([y_train, y_train], ignore_index=True)

    return X_augmented, y_augmented
