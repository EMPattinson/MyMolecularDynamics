import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

data = np.loadtxt("./MMD.csv", skiprows=1, delimiter=",")

Nequil = 1000
block_size = 1

print("TEMPERATURE DENSITY E_tot E_kin E_pot")
print(str(round(np.mean(data[int(Nequil*0.1):Nequil,1]),5)))
print(str(round(np.mean(data[Nequil:,1]),5)),str(round(np.mean(data[Nequil:,2]),3)),str(round(np.mean(data[Nequil:,3]),3)),str(round(np.mean(data[Nequil:,4]),3)),str(round(np.mean(data[Nequil:,5]),3)))

timestep = data[:math.floor(len(data[:,0])/block_size) * block_size,0].reshape((-1, block_size))
timestep = np.mean(timestep, axis = 1)
Epot = data[:math.floor(len(data[:,0])/block_size) * block_size,5].reshape((-1, block_size))
Epot = np.mean(Epot, axis = 1)
Ekin = data[:math.floor(len(data[:,0])/block_size) * block_size,4].reshape((-1, block_size))
Ekin = np.mean(Ekin, axis = 1)
Etot = data[:math.floor(len(data[:,0])/block_size) * block_size,3].reshape((-1, block_size))
Etot = np.mean(Etot, axis = 1)

matplotlib.rcParams.update({'font.size': 22})
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
fig, axs = plt.subplots(3,1,figsize=(1338*px, 761*px), sharex=True)
# ax.plot(data[:,0], data[:,1], label="Temp")
# ax.plot(data[:,0], data[:,2], label="density")
axs[0].plot(timestep, Epot, label="MMD")
axs[1].plot(timestep, Ekin)
# axs[2].plot(timestep, (Etot-Etot[0])/Etot[0])
axs[2].plot(timestep, Etot)

axs[3].set_xlabel("Timestep")
axs[0].set_ylabel("$E_{potenital}$")
axs[1].set_ylabel("$E_{kinetic}$")
axs[2].set_ylabel("$E_{total}$")
axs[0].legend()

plt.show()