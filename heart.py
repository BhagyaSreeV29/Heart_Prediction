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
    A modular pipeline class for preprocessing, training, and evaluating a machine learning model
    on a heart disease dataset.
    """

    def __init__(self, model: Optional[ClassifierMixin] = None, data_path: str = 'Heart_Disease_Prediction.csv') -> None:
        """
        Initializes the HeartDiseaseModel class.

        Parameters:
        - model: Optional sklearn classifier, defaults to Logistic Regression if None
        - data_path: Path to the CSV dataset
        """
        self.model: ClassifierMixin = model if model else LogisticRegression()
        self.data_path: str = data_path
        self.scaler: StandardScaler = StandardScaler()

    def get_password(self) -> Optional[str]:
        """
        Loads and returns a password stored in an environment variable.

        Returns:
        - Password as a string if available, otherwise None
        """
        load_dotenv()
        return os.getenv("PASSWORD")

    def load_data(self) -> DataFrame:
        """
        Loads the dataset from the specified CSV file.

        Returns:
        - Pandas DataFrame containing the dataset
        """
        return pd.read_csv(self.data_path)

    def convert_dtypes(self, df: DataFrame) -> DataFrame:
        """
        Converts specific columns to categorical datatypes.

        Parameters:
        - df: Input DataFrame

        Returns:
        - DataFrame with converted data types
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
        Removes rows with outliers using the IQR method.

        Parameters:
        - df: Input DataFrame

        Returns:
        - Cleaned DataFrame with outliers removed
        """
        num_df = df.select_dtypes(include='number')
        Q1 = num_df.quantile(0.25)
        Q3 = num_df.quantile(0.75)
        IQR = Q3 - Q1
        return df[~((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).any(axis=1)]

    def encode_and_scale(self, df: DataFrame) -> Tuple[DataFrame, Series, DataFrame, Series]:
        """
        Encodes categorical variables, scales features, and splits data into train and test sets.

        Parameters:
        - df: Input DataFrame

        Returns:
        - X_train_scaled: Scaled training features
        - y_train: Training labels
        - X_test_scaled: Scaled test features
        - y_test: Test labels
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
        Trains the classifier and evaluates it using classification metrics.

        Parameters:
        - X_train: Scaled training features
        - y_train: Training labels
        - X_test: Scaled test features
        - y_test: Test labels
        """
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

    def run_pipeline(self) -> None:
        """
        Executes the complete data processing and model training pipeline.
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