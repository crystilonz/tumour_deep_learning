from utils.text_model_evaluate import pickle_test_lstm

if __name__ == '__main__':
    pickle_test_lstm(input_size=1024,
                     embed_size=1024,
                     hidden_size=1024,
                     num_layers=1,
                     vocab_path="/Users/muang/PycharmProjects/tumour_deep_learning/src/datasets/lung_text/vocab.json",
                     pickle_path="/Users/muang/PycharmProjects/tumour_deep_learning/src/datasets/lung_text/h_vec.pkl",
                     model_path="/Users/muang/PycharmProjects/tumour_deep_learning/src/dist_models/LSTM_h_vec/model.pt",
                     )