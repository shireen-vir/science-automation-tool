import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Loads data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def split_data(data, target_column, test_size=0.2, random_state=42):
    """
    Splits data into training and testing sets.

    Args:
        data (pd.DataFrame): Data to split.
        target_column (str): Name of the target column.
        test_size (float, optional): Proportion of data to use for testing. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: Training data, testing data
    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def main():
    """
    Main function for the science-automation-tool.
    """
    file_path = 'data.csv'
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = split_data(data, 'target')
    print("Data loaded and split successfully.")

if __name__ == "__main__":
    main()