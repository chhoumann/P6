from data.get_data import DataType, get_data
from transformer.transformer import Transformer

if __name__ == "__main__":
    print("Starting")
    data = get_data(DataType.delhi_small)

    transformer = Transformer(n_encoder_inputs=8, n_decoder_inputs=9, d_model=512, nhead=8)
    pred = transformer((data["X_train"], data["y_train"]))

    print(pred)

    #print(data["X_train"])
