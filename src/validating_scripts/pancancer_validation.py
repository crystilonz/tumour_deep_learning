from k_fold_validation_pancancer import k_fold_validation_pancancer
from models.pan_cancer_classifier import PanCancerClassifier

if __name__ == '__main__':
    model_lambda = lambda: PanCancerClassifier(input_size=34,
                                hidden_size=42,
                                output_size=10)
    k_fold_validation_pancancer(model_lambda,
                                epochs=30)