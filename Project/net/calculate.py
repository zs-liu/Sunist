import torch
import torch.nn.functional as F


def train(model, loader, optimizer, criterion, epoch, show_frq, batch_size, accu_list, loss_list):
    model.train()
    total_loss = 0
    total_accuracy = 0
    for batch, (data, label) in enumerate(loader):
        optimizer.zero_grad()
        output = model(data)
        # print(output.shape)
        # print(label.shape)
        loss = criterion(output, label)
        total_loss += loss.item()
        pre = output.max(1, keepdim=False)[1]
        total_accuracy += pre.eq(label.view_as(pre)).sum().item() / 500 / 500
        print(pre.eq(label.view_as(pre)).sum().item() / 500 / 500 / batch_size)
        loss.backward()
        optimizer.step()
        if (batch + 1) % show_frq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage Loss: {:.4f}\tAverage Accuracy: {:.2f}%'
                  .format(epoch, (batch + 1) * batch_size,
                          len(loader.dataset),
                          100. * (batch + 1) * batch_size / len(loader.dataset),
                          total_loss / show_frq / batch_size,
                          total_accuracy / show_frq / batch_size * 100))
            loss_list.append(total_loss / show_frq / batch_size)
            accu_list.append(total_accuracy / show_frq / batch_size)
            total_loss = 0
            total_accuracy = 0


def adjust_learning_rate(optimizer, epoch, lr, freq):
    if freq >= 1 and epoch >= freq // 2:
        lr = lr * (0.1 ** ((epoch - freq // 2) / freq))
    else:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
