import torch
import torch.nn as nn

from data_manipulation.pancancer_from_csv import get_data_from_csv
from models.pancancer_classifier import PanCancerClassifier
from train_pancancer import DEFAULT_SAVED_MODEL_DIR, DEFAULT_SAVED_MODEL_NAME, train_panCancer, DEFAULT_DATA_DIR


def eval_panCancer():
    pan_cancer_model = PanCancerClassifier(input_size=33,
                                           hidden_size=42,
                                           output_size=10)
    try:
        pan_cancer_model.load_state_dict(torch.load(DEFAULT_SAVED_MODEL_DIR + DEFAULT_SAVED_MODEL_NAME))
    except FileNotFoundError:
        # file not found --> retrain
        pan_cancer_model = train_panCancer()

    # device agnostic
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    samples, slides, data, labels = get_data_from_csv(DEFAULT_DATA_DIR)

    # move model to device
    pan_cancer_model.to(device)

    # change to tensor
    data_tensor = torch.tensor(data, dtype=torch.float, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)






