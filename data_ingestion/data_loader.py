import pandas as pd
import logging
import os

# Configure logging operations
logging.basicConfig(filename='development_logs.log', level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(message)s')

class DataLoad:
    """ This class is used to fetch data for training.
    Author: Vikash chauhan 😊
    """

    def __init__(self, dataset: str):
        """
        Constructor to initialize the dataset path.
        
        Parameters:
        -----------
        dataset : str
            Path to the dataset (can be CSV, Excel, etc.).
        """
        self.dataset = dataset

    def fetch_data(self):
        """ 
        Description: This method reads data from the source and returns a pandas DataFrame.
        It raises an exception if it fails.

        Parameters:
        -----------
        dataset: dataset path (supports CSV, Excel).

        Returns:
        --------
        pd.DataFrame : The dataset as a pandas DataFrame if successfully loaded.
        """
        logging.info('Entered the "fetch_data" method of the "DataLoad" class.')  # logging operation

        try:
            if not os.path.exists(self.dataset):
                raise FileNotFoundError(f"The file {self.dataset} does not exist.")
            
            # Check file extension and load accordingly
            file_extension = self.dataset.split('.')[-1].lower()

            if file_extension == 'csv':
                df = pd.read_csv(self.dataset)  # Read CSV file
            elif file_extension in ['xls', 'xlsx']:
                df = pd.read_excel(self.dataset)  # Read Excel file
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # Check if the dataset is empty
            if df.empty:
                raise ValueError("The dataset is empty.")

            logging.info(f'Data loaded successfully. Shape of the data is {df.shape}')  # logging operation
            logging.info('Exited the "fetch_data" method of the "DataLoad" class.')
            return df

        except FileNotFoundError as fnf_error:
            logging.error(f"FileNotFoundError occurred: {fnf_error}")
            logging.info('Data fetching unsuccessful. Exited the "fetch_data" method of the "DataLoad" class.')
            return None

        except ValueError as val_error:
            logging.error(f"ValueError occurred: {val_error}")
            logging.info('Data fetching unsuccessful. Exited the "fetch_data" method of the "DataLoad" class.')
            return None

        except Exception as e:
            logging.error(f"Exception occurred in fetch_data method: {e}")
            logging.info('Data fetching unsuccessful. Exited the "fetch_data" method of the "DataLoad" class.')
            return None

