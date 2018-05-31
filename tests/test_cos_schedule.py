from dlutils.training.scheduler import CosineAnnealingSchedule


def test_cos_schedule():
    '''
    '''
    epoch_max = 100
    scheduler = CosineAnnealingSchedule(
        lr_max=0.1, lr_min=0.01, epoch_max=epoch_max, reset_decay=2.)

    lrs = [scheduler(epoch, 0.1) for epoch in xrange(3 * epoch_max)]

    import matplotlib.pyplot as plt
    plt.plot(lrs)
    plt.show()

    assert lrs[0] == 0.1
    assert abs(lrs[-1] - 0.01) <= 1e-3


if __name__ == '__main__':
    test_cos_schedule()
