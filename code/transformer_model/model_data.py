import torch
import numpy as np
import settings


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]

    input = torch.stack(torch.stack([item[0] for item in data]).chunk(settings.input_window, 1))  # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(settings.input_window, 1))

    return input, target


# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = np.append(input_data[i:i + tw][:-settings.output_window], settings.output_window * [0])
        train_label = input_data[i:i + tw]
        # train_label = input_data[i+settings.output_window:i+tw+settings.output_window]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)


def get():
    time = np.arange(0, 400, 0.1)
    amplitude = np.sin(time) + np.sin(time * 0.05) + np.sin(time * 0.12) * np.random.normal(-0.2, 0.2, len(time))

    # from pandas import read_csv
    # series = read_csv('daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # amplitude = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).reshape(-1)
    amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)

    samples = 2800
    train_data = amplitude[:samples]
    test_data = amplitude[samples:]

    # convert our train data into a pytorch train tensor
    # train_tensor = torch.FloatTensor(train_data).view(-1)
    # todo: add comment.. 
    train_sequence = create_inout_sequences(train_data, settings.input_window)
    train_sequence = train_sequence[:-settings.output_window]  # todo: fix hack?

    # test_data = torch.FloatTensor(test_data).view(-1) 
    test_data = create_inout_sequences(test_data, settings.input_window)
    test_data = test_data[:-settings.output_window]  # todo: fix hack?

    return train_sequence.to(settings.device), test_data.to(settings.device)
