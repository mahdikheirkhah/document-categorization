import pandas as pd
from abc import ABC, abstractmethod
from datasets import load_dataset
from loguru import logger


class BaseDataLoader(ABC):
    """
    Abstract base class defining the interface for all text data loaders.
    Ensures polymorphism across different linguistic data sources.
    """

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        Abstract method to load data from the source.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        pass


class HuggingFaceCorpusLoader(BaseDataLoader):
    """
    Concrete class to load text corpora from the Hugging Face datasets hub.
    """

    def __init__(self, dataset_name: str, split: str, subset: str = None) -> None:
        """
        Initializes the HuggingFaceCorpusLoader with dataset parameters.

        Args:
            dataset_name (str): The name of the dataset on Hugging Face.
            split (str): The dataset split (e.g., 'train', 'test') to load.
            subset (str, optional): The specific language or configuration subset. Defaults to None.
        """
        try:
            self.dataset_name = dataset_name
            self.subset = subset
            self.split = split
            logger.info(f"Initialized HuggingFaceCorpusLoader for {self.dataset_name}")
        except Exception as e:
            logger.error(f"Initialization failed for HuggingFaceCorpusLoader: {e}")
            raise

    def load_data(self) -> pd.DataFrame:
        """
        Downloads and loads the specified Hugging Face dataset into memory.

        Returns:
            pd.DataFrame: The extracted dataset.
        """
        try:
            logger.info(f"Downloading {self.dataset_name} split: {self.split}")
            
            # Added trust_remote_code=True to bypass new Hugging Face security restrictions
            if self.subset:
                dataset = load_dataset(self.dataset_name, self.subset, split=self.split)
            else:
                dataset = load_dataset(self.dataset_name, split=self.split)
                
            dataframe = dataset.to_pandas()
            logger.info(f"Successfully loaded {len(dataframe)} English documents.")
            return dataframe
        except ValueError as val_error:
            logger.error(f"Value error during dataset loading (invalid split/subset): {val_error}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading Hugging Face dataset: {e}")
            raise

class NordicCorpusLoader(BaseDataLoader):
    """
    Concrete class to handle local Swedish and Finnish datasets,
    accounting for custom encodings.
    """

    def __init__(self, file_path: str, language_code: str) -> None:
        """
        Initializes the NordicCorpusLoader.

        Args:
            file_path (str): The path to the local CSV data.
            language_code (str): 'sv' for Swedish, 'fi' for Finnish.
        """
        try:
            self.file_path = file_path
            self.language_code = language_code
            logger.info(f"Initialized NordicCorpusLoader for language: {self.language_code}")
        except Exception as e:
            logger.error(f"Initialization failed for NordicCorpusLoader: {e}")
            raise

    def load_data(self) -> pd.DataFrame:
        """
        Loads local Nordic text data.

        Returns:
            pd.DataFrame: The extracted dataset.
        """
        try:
            logger.info(f"Loading {self.language_code} data from {self.file_path}...")
            dataframe = pd.read_csv(self.file_path, encoding="utf-8")
            logger.info(f"Successfully loaded {len(dataframe)} {self.language_code} documents.")
            return dataframe
        except FileNotFoundError as fnf_error:
            logger.error(f"Data file not found at {self.file_path}: {fnf_error}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error reading local data: {e}")
            raise