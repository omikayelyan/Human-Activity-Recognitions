import pandas as pd


def load_and_prepare_data(train_csv: str, test_csv: str):
    """
    Loads train and test CSV files, removes missing values from training data,
    and splits data into features (X) and labels (y).

    Parameters:
        train_csv (str): Path to the training CSV file
        test_csv (str): Path to the testing CSV file

    Returns:
        X_train (DataFrame): Training features
        y_train (Series): Training labels
        X_test (DataFrame): Testing features
        y_test (Series): Testing labels
    """

    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)

    train_data.dropna(inplace=True)

    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    return X_train, y_train, X_test, y_test


