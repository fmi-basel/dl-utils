from dlutils.training.scheduler import CosineAnnealingSchedule


def test_cos_schedule():
    '''
    '''
    epoch_max = 100
    scheduler = CosineAnnealingSchedule(
        lr_max=0.1, lr_min=0.0001, epoch_max=100, epoch_max_growth=2, reset_decay=0.1)

    lrs = [scheduler(epoch, 0.1) for epoch in range(3 * epoch_max)]
    
    assert lrs[0] == 0.1
    assert abs(lrs[99] - 0.0001) <= 1e-3
    assert abs(lrs[100] - 0.01)  <= 1e-3
    assert abs(lrs[-1] - 0.0001) <= 1e-3


if __name__ == '__main__':
    test_cos_schedule()
