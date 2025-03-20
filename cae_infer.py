import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from architectures.conv_ae import ConvAutoencoder


def animate_comparison(original, reconstructed):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    img1 = axes[0, 0].imshow(original[0, 0], cmap='gray', vmin=original.min(), vmax=original.max())
    img2 = axes[1, 0].imshow(original[0, 1], cmap='gray', vmin=original.min(), vmax=original.max())
    img3 = axes[0, 1].imshow(reconstructed[0, 0], cmap='gray', vmin=reconstructed.min(), vmax=reconstructed.max())
    img4 = axes[1, 1].imshow(reconstructed[0, 1], cmap='gray', vmin=reconstructed.min(), vmax=reconstructed.max())
    errx = np.abs(original[0, 0] - reconstructed[0, 0])
    erry = np.abs(original[0, 1] - reconstructed[0, 1])
    img5 = axes[0, 2].imshow(errx, cmap='gray', vmin=errx.min(), vmax=errx.max() )
    img6 = axes[1, 2].imshow(erry, cmap='gray', vmin=erry.min(), vmax=erry.max() )
    
    axes[0, 0].set_title("Original Channel 1")
    axes[1, 0].set_title("Original Channel 2")
    axes[0, 1].set_title("Reconstructed Channel 1")
    axes[1, 1].set_title("Reconstructed Channel 2")
    axes[0, 2].set_title("Abs error Channel 1")
    axes[1, 2].set_title("Abs error Channel 2")
    
    
    def update(frame):
        img1.set_array(original[frame, 0])
        img2.set_array(original[frame, 1])
        img3.set_array(reconstructed[frame, 0])
        img4.set_array(reconstructed[frame, 1])
        img5.set_array( np.abs(original[frame, 0] - reconstructed[frame, 0]) )
        img6.set_array( np.abs(original[frame, 0] - reconstructed[frame, 0]) )
        return img1, img2, img3, img4, img5, img6
        
    ani = animation.FuncAnimation(fig, update, frames=original.shape[0], interval=50, blit=False)
    plt.show()

def recon_in_batches(ae, data, batch_size=256):
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    encoded_batches = []
    with torch.no_grad():  # No need to track gradients
        for batch in dataloader:
            encoded_batches.append(ae(batch))  # Encode batch-wise
    
    return torch.cat(encoded_batches, dim=0)  # Concatenate results


save_loc = "models/cae.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device {device}")

data = np.load('data/data_a_05.npy').astype(np.float32)
data = data[100:,]
input_tensor = torch.from_numpy(data).to(device)
# dataloader = DataLoader(data, batch_size=512, shuffle=False)
# print(data.shape,data.min(),data.max(),data.dtype)
# torch.cuda.empty_cache()

model = ConvAutoencoder(latent_dim=256).to(device)
model.load_state_dict(torch.load(save_loc, weights_only=True))
model.eval()

print(f"Model loaded from {save_loc}")

output = recon_in_batches(model, input_tensor, batch_size=256)
animate_comparison(input_tensor.detach().cpu(),output.detach().cpu())
