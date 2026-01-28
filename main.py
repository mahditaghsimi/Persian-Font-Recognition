from make_data.make_data import make_data
from augmentations.augmentations import process_dataset

if __name__ == "__main__":
    make_data()
    process_dataset()
