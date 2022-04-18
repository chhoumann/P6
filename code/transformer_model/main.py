import torch
import numpy as np
import time
import math
import settings
import model_plot
import model_data

from transformer_model import TransformerAttentionModel

torch.manual_seed(0)
np.random.seed(0)

train_data, val_data = model_data.get()
model = TransformerAttentionModel().to(settings.device)

# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.98)

best_val_loss = float("inf")
best_model = None


def train():
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, settings.batch_size)):
        data, targets = model_data.get_batch(train_data, i, settings.batch_size)
        optimizer.zero_grad()
        output = model(data)

        if settings.calculate_loss_over_all_values:
            loss = settings.criterion(output, targets)
        else:
            loss = settings.criterion(output[-settings.output_window:], targets[-settings.output_window:])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / settings.batch_size / 5)

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // settings.batch_size, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


for epoch in range(1, settings.epochs + 1):
    epoch_start_time = time.time()
    model.train()

    if epoch % 10 == 0:
        val_loss = model_plot.plot_and_loss(model, val_data, epoch)
        model.predict_future(val_data, 200)
    else:
        val_loss = model.evaluate(val_data)

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (
            time.time() - epoch_start_time),
                                                                                                  val_loss,
                                                                                                  math.exp(
                                                                                                      val_loss)))
    print('-' * 89)

    # if val_loss < best_val_loss:
    #    best_val_loss = val_loss
    #    best_model = model

    scheduler.step()

    # src = torch.rand(settings.input_window, batch_size, 1) # (source sequence length,batch size,feature number) 
    # out = model(src)
    #
    # print(out)
    # print(out.shape)
