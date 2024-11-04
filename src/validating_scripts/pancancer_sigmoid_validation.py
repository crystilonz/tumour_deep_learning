from k_fold_validation_pancancer import k_fold_validation_pancancer
from models.pan_cancer_sigmoid import SigmoidalPanCancerClassifier

if __name__ == '__main__':
    model_lambda = lambda: SigmoidalPanCancerClassifier(input_size=34,
                                                        hidden_size=42,
                                                        output_size=10)

    k_fold_validation_pancancer(model_lambda,
                                epochs=30,
                                results_dir="sigmoidal_pancancer",
                                best_model_save_name="sigmoidal_pancancer_checkpoint", )
