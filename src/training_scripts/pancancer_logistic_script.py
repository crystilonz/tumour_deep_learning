from train_pancancer import train_pan_cancer
from models.logistic_regression import PanCancerLogisticRegression

if __name__ == '__main__':
    model = PanCancerLogisticRegression(input_size=34,
                                        output_size=10)
    train_pan_cancer(model,
                     epochs=40,
                     save_dir="logistic_regression",
                     save_name="logistic_regression_checkpoints",)