from train_pancancer import train_pan_cancer
from models.pan_cancer_leaky_with_dropout import LeakyPanCancerClassifierWithDropout

if __name__ == '__main__':
    model = LeakyPanCancerClassifierWithDropout(input_size=34,
                                                hidden_size=42,
                                                output_size=10)
    train_pan_cancer(model,
                     epochs=30,
                     save_dir="leaky_pancancer_dropout",
                     save_name="leaky_pancancer_dropout_checkpoints",)