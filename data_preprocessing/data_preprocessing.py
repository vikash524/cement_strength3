import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Configuring logging operations
logging.basicConfig(filename='development_logs.log', level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(message)s')


class DataPreprocessor:
    """This class is used to preprocess the data for modeling.
    
    Author: vikash chauhan 😊
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        A pandas dataframe that has to be preprocessed
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def rem_outliers(self, column_name: str) -> pd.DataFrame:
        """Removes outliers from the specified column using the Interquartile Range (IQR) method.
        
        Parameters
        ----------
        column_name : str
            Column name for which the outliers need to be removed.

        Returns
        -------
        pd.DataFrame
            DataFrame with outliers removed from the specified column.
        """
        logging.info('Entered the "rem_outliers" method of the "DataPreprocessor" class.')

        try:
            q1 = self.dataframe[column_name].quantile(0.25)
            q3 = self.dataframe[column_name].quantile(0.75)
            iqr = q3 - q1
            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr

            # Remove outliers
            self.dataframe = self.dataframe[(self.dataframe[column_name] >= lower_limit) &
                                            (self.dataframe[column_name] <= upper_limit)]
            
            logging.info(f'Outlier treatment using IQR method: Outliers removed from {column_name}. '
                         f'New shape of data is {self.dataframe.shape}.')
            logging.info('Exited the rem_outliers method of the DataPreprocessor class.')
            return self.dataframe

        except Exception as e:
            logging.error(f"Exception occurred in rem_outliers method: {str(e)}")
            logging.info('Outlier removal unsuccessful. Exited the rem_outliers method.')
            return self.dataframe

    def data_split(self, test_size: float, stratify: bool = False) -> tuple:
        """Splits the DataFrame into train and test sets.
        
        Parameters
        ----------
        test_size : float
            Percentage of the data to be used as the test set (between 0 and 1).
        stratify : bool, optional
            Whether to stratify the split based on the target variable (default is False).
        
        Returns
        -------
        tuple
            A tuple containing the train and test DataFrames.
        """
        logging.info('Entered the data_split method of the DataPreprocessor class.')

        try:
            # Stratified split for classification (if applicable)
            if stratify:
                y = self.dataframe.iloc[:, -1]  # Assuming the last column is the target variable
                df_train, df_test = train_test_split(self.dataframe, test_size=test_size, shuffle=True, random_state=42, stratify=y)
            else:
                df_train, df_test = train_test_split(self.dataframe, test_size=test_size, shuffle=True, random_state=42)

            logging.info(f'Train-test split successful. Shapes - train: {df_train.shape}, test: {df_test.shape}.')
            logging.info('Exited the data_split method of the DataPreprocessor class.')
            return df_train, df_test

        except Exception as e:
            logging.error(f"Exception occurred in data_split method: {str(e)}")
            logging.info('Train-test split unsuccessful. Exited the data_split method.')
            return None, None

    def feature_scaling(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple:
        """Scales the features of the train and test sets using StandardScaler.
        
        Parameters
        ----------
        df_train : pd.DataFrame
            Training DataFrame.
        df_test : pd.DataFrame
            Test DataFrame.

        Returns
        -------
        tuple
            Scaled train and test DataFrames.
        """
        logging.info('Entered the feature_scaling method of the DataPreprocessor class.')

        try:
            scaler = StandardScaler()
            columns = df_train.columns  # Store column names for re-creating DataFrame

            # Apply scaling
            df_train_scaled = scaler.fit_transform(df_train)
            df_test_scaled = scaler.transform(df_test)

            # Convert numpy arrays back to DataFrame
            df_train_scaled = pd.DataFrame(df_train_scaled, columns=columns)
            df_test_scaled = pd.DataFrame(df_test_scaled, columns=columns)

            logging.info('Feature scaling successful for both train and test datasets.')
            logging.info('Exited the feature_scaling method of the DataPreprocessor class.')

            return df_train_scaled, df_test_scaled

        except Exception as e:
            logging.error(f"Exception occurred in feature_scaling method: {str(e)}")
            logging.info('Feature scaling unsuccessful. Exited the feature_scaling method.')
            return df_train, df_test

    def splitting_as_x_y(self, df_train: pd.DataFrame, df_test: pd.DataFrame, column_name: str) -> tuple:
        """Splits the data into independent (X) and dependent (y) variables for both train and test sets.
        
        Parameters
        ----------
        df_train : pd.DataFrame
            Training DataFrame.
        df_test : pd.DataFrame
            Test DataFrame.
        column_name : str
            The name of the target column.

        Returns
        -------
        tuple
            Tuple containing X_train, y_train, X_test, and y_test.
        """
        logging.info('Entered the splitting_as_x_y method of the DataPreprocessor class.')

        try:
            X_train = df_train.drop(column_name, axis=1)
            y_train = df_train[column_name]
            X_test = df_test.drop(column_name, axis=1)
            y_test = df_test[column_name]

            logging.info(f'Splitting successful. Shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, '
                         f'X_test: {X_test.shape}, y_test: {y_test.shape}.')
            logging.info('Exited the splitting_as_x_y method of the DataPreprocessor class.')

            return X_train, y_train, X_test, y_test

        except Exception as e:
            logging.error(f"Exception occurred in splitting_as_x_y method: {str(e)}")
            logging.info('Data splitting into X and y unsuccessful. Exited the splitting_as_x_y method.')
            return None, None, None, None

