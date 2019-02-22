import matplotlib.pyplot as plt


def plot_loss_and_acc(loss_and_acc_dict):
    fig = plt.figure()
    tmp = list(loss_and_acc_dict.values())
    max_epoch = len(tmp[0][0])

    _max_loss = max([max(x[0]) for x in loss_and_acc_dict.values()])
    _min_loss = min([min(x[0]) for x in loss_and_acc_dict.values()])

    max_loss = _max_loss + (_max_loss - _min_loss) * 0.1
    min_loss = max(0, _min_loss - (_max_loss - _min_loss) * 0.1)

    for name, lossAndAcc in loss_and_acc_dict.items():
        plt.plot(range(1, 1 + max_epoch), lossAndAcc[0], '-s', label=name)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(0, max_epoch + 1, 2))
    plt.axis([0, max_epoch + 1, min_loss, max_loss])
    plt.legend()
    plt.savefig("./loss.png")
    plt.show()

    _max_accu = max([max(x[1]) for x in loss_and_acc_dict.values()])
    _min_accu = min([min(x[1]) for x in loss_and_acc_dict.values()])

    max_accu = min(1, _max_accu + (_max_accu - _min_accu) * 0.1)
    min_accu = max(0, _min_accu - (_max_accu - _min_accu) * 0.1)

    fig = plt.figure()

    for name, lossAndAcc in loss_and_acc_dict.items():
        plt.plot(range(1, 1 + max_epoch), lossAndAcc[1], '-s', label=name)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(range(0, max_epoch + 1, 2))
    plt.axis([0, max_epoch + 1, min_accu, max_accu])
    plt.legend()
    plt.savefig("./accu.png")
    plt.show()
