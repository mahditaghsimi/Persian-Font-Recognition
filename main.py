from make_data.make_data import make_data
from augmentations.augmentations import process_dataset
from model.model import train_model
from figures.evaluate import evaluate_model



if __name__ == "__main__":
    make_data()
    process_dataset()
    train_model(data_dir='./data_augmentations')
    evaluate_model()