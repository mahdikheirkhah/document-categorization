import pytest
from utils.data_loader import HuggingFaceCorpusLoader, NordicCorpusLoader

def test_huggingface_loader_invalid_dataset() -> None:
    """
    Tests the exception flow of the HuggingFaceCorpusLoader when given a non-existent dataset.
    """
    try:
        loader = HuggingFaceCorpusLoader(dataset_name="non_existent_fake_dataset", subset="en", split="train")
        loader.load_data()
    except Exception as e:
        assert isinstance(e, ValueError) or type(e).__name__ == "DatasetNotFoundError"

def test_nordic_loader_file_not_found() -> None:
    """
    Tests the exception flow for the NordicCorpusLoader when a file does not exist.
    """
    try:
        loader = NordicCorpusLoader(file_path="fake_path.csv", language_code="sv")
        loader.load_data()
    except Exception as e:
        assert isinstance(e, FileNotFoundError)