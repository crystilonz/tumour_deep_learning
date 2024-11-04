from k_fold_validation_pancancer import k_fold_validation_pancancer
from models.pan_cancer_leaky import LeakyPanCancerClassifier

if __name__ == '__main__':
    model_lamda = lambda: LeakyPanCancerClassifier(input_size=34,
                                                   hidden_size=42,
                                                   output_size=10)
    k_fold_validation_pancancer(model_lamda,
                                epochs=30,
                                results_dir="leaky_pancancer",
                                best_model_save_name="leaky_pancancer_checkpoints")
