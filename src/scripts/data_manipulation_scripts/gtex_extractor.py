from data_manipulation.pancancer_csv_from_raw import build_csv_from_raw

if __name__ == '__main__':
    build_csv_from_raw(leiden_csv='/Users/muang/PycharmProjects/tumour_deep_learning/data/GTEX_cohort/GTEX_he_combined_leiden_2p0__fold1.csv',
                       classes_csv='/Users/muang/PycharmProjects/tumour_deep_learning/data/GTEX_cohort/classes.csv',
                       save_to='../../datasets/gtex_pancancer/gtex.csv',
                       name_format='None')