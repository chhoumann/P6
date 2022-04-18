from data.get_data import DataType, get_data


if __name__ == "__main__":
    print("Starting")
    data = get_data(DataType.delhi_small)
    print(data["X_train"])
