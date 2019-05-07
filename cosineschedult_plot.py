from dlutils.training.scheduler import CosineAnnealingSchedule
import matplotlib.pyplot as plt

from numpy import ceil

epoch_max_growth = 2
n_restarts = 3
epochs = 2000

n_restarts_factor = sum(epoch_max_growth**x for x in range(n_restarts))
epochs_to_restart = (epochs + 1) / n_restarts_factor
if epochs_to_restart < 1:
    raise ValueError(
        'Initial epoch_to_restart ({}) < 1. Decrease n_restarts ({}) or epoch_to_restart_growth ({})'.format(epochs_to_restart, n_restarts, epoch_max_growth))
epochs_to_restart = int(ceil(epochs_to_restart))

schedule = CosineAnnealingSchedule(lr_max=0.001, lr_min=0.000001, epoch_max=epochs_to_restart, epoch_max_growth=epoch_max_growth, reset_decay=1.0)


x = list(range(epochs))
y = [schedule(i) for i in x]
plt.plot(x,y, )
plt.yscale('log')
plt.show()
