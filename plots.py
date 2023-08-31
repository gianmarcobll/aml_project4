import numpy as np
import matplotlib.pyplot as plt
x = []
y = []

with np.load('training/logs/LR_0_source_source/evaluations.npz') as data:
    x.append(data['timesteps'])
    y.append(np.mean(data['results'], axis=1))
with np.load('training/logs/LR_1_source_source/evaluations.npz') as data:
    x.append(data['timesteps'])
    y.append(np.mean(data['results'], axis=1))
with np.load('training/logs/LR_2_source_source/evaluations.npz') as data:
    x.append(data['timesteps'])
    y.append(np.mean(data['results'], axis=1))
with np.load('training/logs/LR_3_source_source/evaluations.npz') as data:
    x.append(data['timesteps'])
    y.append(np.mean(data['results'], axis=1))
with np.load('training/logs/LR_4_source_source/evaluations.npz') as data:
    x.append(data['timesteps'])
    y.append(np.mean(data['results'], axis=1))
plt.title("Source-Source")
plt.xlabel("Training steps")
plt.ylabel("Mean reward")
plt.plot(x[0], y[0])
plt.plot(x[1], y[1])
plt.plot(x[2], y[2])
plt.plot(x[3], y[3])
plt.plot(x[4], y[4])
plt.legend(["lr = 1e-3", "lr = 3e-4", "lr = 1e-4", "lr = 1e-5", "lr = 1e-6"])
plt.show()

x = []
y = []

with np.load('training/logs/LR_0_source_target/evaluations.npz') as data:
    x.append(data['timesteps'])
    y.append(np.mean(data['results'], axis=1))
with np.load('training/logs/LR_1_source_target/evaluations.npz') as data:
    x.append(data['timesteps'])
    y.append(np.mean(data['results'], axis=1))
with np.load('training/logs/LR_2_source_target/evaluations.npz') as data:
    x.append(data['timesteps'])
    y.append(np.mean(data['results'], axis=1))
with np.load('training/logs/LR_3_source_target/evaluations.npz') as data:
    x.append(data['timesteps'])
    y.append(np.mean(data['results'], axis=1))
with np.load('training/logs/LR_4_source_target/evaluations.npz') as data:
    x.append(data['timesteps'])
    y.append(np.mean(data['results'], axis=1))
plt.title("Source-Target")
plt.xlabel("Training steps")
plt.ylabel("Mean reward")
plt.plot(x[0], y[0])
plt.plot(x[1], y[1])
plt.plot(x[2], y[2])
plt.plot(x[3], y[3])
plt.plot(x[4], y[4])
plt.legend(["lr = 1e-3", "lr = 3e-4", "lr = 1e-4", "lr = 1e-5", "lr = 1e-6"])
plt.show()

x = []
y = []

with np.load('training/logs/LR_0_target_target/evaluations.npz') as data:
    x.append(data['timesteps'])
    y.append(np.mean(data['results'], axis=1))
with np.load('training/logs/LR_1_target_target/evaluations.npz') as data:
    x.append(data['timesteps'])
    y.append(np.mean(data['results'], axis=1))
with np.load('training/logs/LR_2_target_target/evaluations.npz') as data:
    x.append(data['timesteps'])
    y.append(np.mean(data['results'], axis=1))
with np.load('training/logs/LR_3_target_target/evaluations.npz') as data:
    x.append(data['timesteps'])
    y.append(np.mean(data['results'], axis=1))
with np.load('training/logs/LR_4_target_target/evaluations.npz') as data:
    x.append(data['timesteps'])
    y.append(np.mean(data['results'], axis=1))
plt.title("Target-Target")
plt.xlabel("Training steps")
plt.ylabel("Mean reward")
plt.plot(x[0], y[0])
plt.plot(x[1], y[1])
plt.plot(x[2], y[2])
plt.plot(x[3], y[3])
plt.plot(x[4], y[4])
plt.legend(["lr = 1e-3", "lr = 3e-4", "lr = 1e-4", "lr = 1e-5", "lr = 1e-6"])
plt.show()
