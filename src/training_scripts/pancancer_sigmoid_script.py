from train_pancancer import train_pan_cancer
from models.pan_cancer_sigmoid import SigmoidalPanCancerClassifier
if __name__ == '__main__':
    model = SigmoidalPanCancerClassifier(input_size=34,
                                     hidden_size=42,
                                     output_size=10)

    train_pan_cancer(model,
                     epochs=30,
                     save_dir="sigmoidal_pancancer",
                     save_name="sigmoidal_pancancer_checkpoint",)
