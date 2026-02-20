import pandas as pd
from sklearn.preprocessing import StandardScaler


def augment_with_scaled_data(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Performs data augmentation by creating a scaled version of the training features
    and appending it to the original training data.

    Parameters:
        X_train (DataFrame): Original training features
        y_train (Series): Original training labels

    Returns:
        X_augmented (DataFrame): Original + scaled features
        y_augmented (Series): Original + duplicated labels
    """

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_train)
    X_scaled = pd.DataFrame(X_scaled, columns=X_train.columns)

    X_augmented = pd.concat([X_train, X_scaled], axis=0).reset_index(drop=True)

    y_augmented = pd.concat([y_train, y_train], axis=0).reset_index(drop=True)

    return X_augmented, y_augmented
