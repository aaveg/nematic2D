import torch

import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_frames(data):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    img1 = axes[0].imshow(data[0, 0], cmap='gray', vmin=data.min(), vmax=data.max())
    img2 = axes[1].imshow(data[0, 1], cmap='gray', vmin=data.min(), vmax=data.max())
    axes[0].set_title("Channel 1")
    axes[1].set_title("Channel 2")
    
    def update(frame):
        img1.set_array(data[frame, 0])
        img2.set_array(data[frame, 1])
        return img1, img2
    
    ani = animation.FuncAnimation(fig, update, frames=data.shape[0], interval=100, blit=False)
    plt.show()


data = np.load('data/data_a_08.npy')[:1000,:]
print(data.shape,data.min(),data.max())

animate_frames(data)


