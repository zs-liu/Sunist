from Transfer import Taker
from Transfer import Processor
from Transfer import Croper
from Transfer import Producer
from net.dataset import SunistDataset
from net.fcn import FCNNet
from net.deeperfcn import FCNNet_deep
from net.calculate import train, adjust_learning_rate
from tools.plot import plot_loss_and_acc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


def _crop():
    croper = Croper(data_input_dir="data/pre_data/", label_input_dir="data/pre_label/",
                    data_output_dir="data/final_data/", label_output_dir="data/final_label/",
                    number=100)
    croper.crop(x_pos=550, y_pos=100, crop_width=500, crop_height=500)


def _train(load_dir="", save_dir=""):
    batch_size = 20
    learning_rate = 0.001
    weight_decay = 0.001
    epochs = 8
    show_frq = 5
    adjust = True

    dataset = SunistDataset(train=True, data_root="data/final_data/", label_root="data/final_label/", number=200)
    train_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    if load_dir != "":
        model = torch.load(load_dir)
    else:
        model = FCNNet()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_accu_list = []
    train_loss_list = []
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer=optimizer, criterion=criterion,
              epoch=epoch, show_frq=show_frq, batch_size=batch_size,
              accu_list=train_accu_list, loss_list=train_loss_list)
        if adjust:  # if learning rate decay is needed
            adjust_learning_rate(optimizer, epoch, learning_rate, freq=3)
    plot_loss_and_acc({'FCNNet': [train_loss_list, train_accu_list]})
    if save_dir != "":
        torch.save(model, save_dir)
    print("Train finish.")


def _use(load_dir=""):
    if load_dir == "":
        print("No moudle!")
        return
    fcn = torch.load(load_dir)
    fcn.eval()
    produce = Producer(input_dir="data/video/180907002cap.mp4", output_dir="data/video/1.avi")
    success, frame = produce.read()
    while success:
        origin = frame[100:600, 620:1120]
        output = frame[100:600, 620:1120]
        output = np.average(output, axis=2).reshape((1, 1, 500, 500))
        output = output.astype(np.float32) / output.max()
        output = fcn(torch.tensor(output))
        output = output.max(1, keepdim=False)[1] * 255
        output = np.array(output).reshape(500, 500)

        origin[output > 0, :] = 255
        frame[100:600, 620:1120] = origin
        '''frame[100:600, 550:1050, 0] = output
        frame[100:600, 550:1050, 1] = output
        frame[100:600, 550:1050, 2] = output'''
        produce.write(frame)
        success, frame = produce.read()
    print("Produce success.")
    produce.release()


def main():
    #_train(load_dir="", save_dir="net/network6.pkl")
    #_train(load_dir="net/network4.pkl", save_dir="net/network4.pkl")
    _use(load_dir="net/network5.pkl")


if __name__ == '__main__':
    main()
