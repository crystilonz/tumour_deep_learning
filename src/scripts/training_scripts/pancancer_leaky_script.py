from train_pancancer import train_pan_cancer
from models.pan_cancer_leaky import LeakyPanCancerClassifier
if __name__ == '__main__':
    model = LeakyPanCancerClassifier(input_size=34,
                                     hidden_size=42,
                                     output_size=10)
    train_pan_cancer(model,
                     epochs=30,
                     save_dir="leaky_pancancer",
                     save_name="leaky_pancancer_checkpoints",)