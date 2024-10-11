from train_pancancer import train_pan_cancer
from models.pan_cancer_classifier import PanCancerClassifier

if __name__ == '__main__':
    model = PanCancerClassifier(input_size=34,
                                hidden_size=42,
                                output_size=10)
    train_pan_cancer(pan_cancer_model=model,
                     epochs=50)