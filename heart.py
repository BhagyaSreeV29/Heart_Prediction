import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from warnings import filterwarnings
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.base import ClassifierMixin
from pandas import DataFrame, Series

filterwarnings('ignore')

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.float_format = '{:.6f}'.format


class HeartDiseaseModel:
    """
    A class to preprocess data, train a model, and evaluate it for heart disease prediction.
    """

    def __init__(self, model: Optional[ClassifierMixin] = None, data_path: str = 'Heart_Disease_Prediction.csv') -> None:
        """
        Initializes the HeartDiseaseModel.

        Args:
            model (Optional[ClassifierMixin]): An optional scikit-learn classifier. Defaults to LogisticRegression.
            data_path (str): Path to the heart disease dataset CSV file.
        """
        self.model: ClassifierMixin = model if model else LogisticRegression()
        self.data_path: str = data_path
        self.scaler: StandardScaler = StandardScaler()

    def get_password(self) -> Optional[str]:
        """
        Loads and retrieves the password from a `.env` file.

        Returns:
            Optional[str]: The password stored under the environment variable 'PASSWORD', or None if not found.
        """
        load_dotenv()
        return os.getenv("PASSWORD")

    def load_data(self) -> DataFrame:
        """
        Loads the dataset from a CSV file.

        Returns:
            DataFrame: A pandas DataFrame containing the dataset.
        """
        return pd.read_csv(self.data_path)

    def convert_dtypes(self, df: DataFrame) -> DataFrame:
        """
        Converts relevant columns to 'category' data type.

        Args:
            df (DataFrame): Raw input DataFrame.

        Returns:
            DataFrame: DataFrame with converted categorical columns.
        """
        cat_cols = [
            'Sex', 'Chest pain type', 'FBS over 120', 'EKG results',
            'Exercise angina', 'Slope of ST', 'Number of vessels fluro',
            'Thallium', 'Heart Disease'
        ]
        for col in cat_cols:
            df[col] = df[col].astype('category')
        return df

    def remove_outliers(self, df: DataFrame) -> DataFrame:
        """
        Removes outliers using the IQR method from all numerical columns.

        Args:
            df (DataFrame): Input DataFrame.

        Returns:
            DataFrame: Cleaned DataFrame with outliers removed.
        """
        num_df = df.select_dtypes(include='number')
        Q1 = num_df.quantile(0.25)
        Q3 = num_df.quantile(0.75)
        IQR = Q3 - Q1
        return df[~((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).any(axis=1)]

    def encode_and_scale(self, df: DataFrame) -> Tuple[DataFrame, Series, DataFrame, Series]:
        """
        Encodes categorical columns, scales features, and splits data into train and test sets.

        Args:
            df (DataFrame): Preprocessed DataFrame.

        Returns:
            Tuple[DataFrame, Series, DataFrame, Series]: X_train, y_train, X_test, y_test datasets.
        """
        df['Heart Disease'] = df['Heart Disease'].replace({'Presence': 1, 'Absence': 0})

        cat_columns = df.select_dtypes(include='category').columns.drop('Heart Disease')
        num_columns = df.select_dtypes(include='int').columns

        encoded = pd.get_dummies(df[cat_columns], drop_first=True)
        df_encoded = pd.concat([df[num_columns], encoded], axis=1)
        df_encoded['Heart Disease'] = df['Heart Disease']

        X = df_encoded.drop('Heart Disease', axis=1)
        y = df_encoded['Heart Disease']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X.columns)
        X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), columns=X.columns)

        return X_train_scaled, y_train, X_test_scaled, y_test

    def train_and_evaluate(self, X_train: DataFrame, y_train: Series, X_test: DataFrame, y_test: Series) -> None:
        """
        Trains the model and prints evaluation metrics on the test set.

        Args:
            X_train (DataFrame): Scaled training features.
            y_train (Series): Training labels.
            X_test (DataFrame): Scaled test features.
            y_test (Series): Test labels.
        """
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

    def run_pipeline(self) -> None:
        """
        Runs the full end-to-end pipeline: load, preprocess, split, train, and evaluate.
        """
        df = self.load_data()
        df = self.convert_dtypes(df)
        df = self.remove_outliers(df)
        X_train, y_train, X_test, y_test = self.encode_and_scale(df)
        self.train_and_evaluate(X_train, y_train, X_test, y_test)


# Run the pipeline
if __name__ == "__main__":
    model_instance = HeartDiseaseModel()
    model_instance.run_pipeline()
