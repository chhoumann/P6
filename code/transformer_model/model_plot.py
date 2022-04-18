import torch
from matplotlib import pyplot
import settings
import model_data


def plot_and_loss(eval_model, data_source, epoch):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = model_data.get_batch(data_source, i, 1)
            # look like the model returns static values for the output window
            output = eval_model(data)
            if settings.calculate_loss_over_all_values:
                total_loss += settings.criterion(output, target).item()
            else:
                total_loss += settings.criterion(output[-settings.output_window:],
                                                 target[-settings.output_window:]).item()

            test_result = torch.cat((test_result, output[-1].view(-1).cpu()),
                                    0)  # todo: check this. -> looks good to me
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

    # test_result = test_result.cpu().numpy()
    len(test_result)

    pyplot.plot(test_result, color="red")
    pyplot.plot(truth[:500], color="blue")
    pyplot.plot(test_result - truth, color="green")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('./graphs/transformer-epoch%d.png' % epoch)
    pyplot.close()

    return total_loss / i
