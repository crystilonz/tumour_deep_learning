from train_pancancer import train_pan_cancer
from models.pan_cancer_classifier_with_dropout import PanCancerClassifierWithDropout

if __name__ == '__main__':
    model = PanCancerClassifierWithDropout(input_size=34,
                                           hidden_size=42,
                                           output_size=10)
    train_pan_cancer(model,
                     epochs=40,
                     save_dir="pancancer_dropout_model",
                     save_name="pancancer_dropout_checkpoints",)
