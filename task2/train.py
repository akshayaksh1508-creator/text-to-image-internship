"""
Task 2: CGAN Training Script
Trains the Conditional GAN for 200 epochs on synthetic shape dataset.
Run: python train.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os, glob, numpy as np, json
from cgan import Generator, Discriminator, NOISE_DIM, NUM_CLASS

class ShapeDataset(Dataset):
    def __init__(self, base_path):
        self.data = []
        for label, cls in enumerate(['circle', 'square', 'triangle']):
            for path in glob.glob(f'{base_path}/{cls}/*.png'):
                self.data.append((path, label))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        path, label = self.data[idx]
        img = torch.tensor(
            np.array(Image.open(path)).astype(np.float32)/127.5-1.0
        ).unsqueeze(0)
        return img, torch.tensor(label)

def train(data_path, ckpt_dir, epochs=200):
    os.makedirs(ckpt_dir, exist_ok=True)
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    loader    = DataLoader(ShapeDataset(data_path), batch_size=64, shuffle=True)
    G         = Generator().to(device)
    D         = Discriminator().to(device)
    opt_G     = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5,0.999))
    opt_D     = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5,0.999))
    criterion = nn.BCELoss()
    g_losses, d_losses = [], []

    for epoch in range(1, epochs+1):
        g_sum = d_sum = 0
        for real_imgs, labels in loader:
            real_imgs = real_imgs.to(device)
            labels    = labels.to(device)
            bs        = real_imgs.size(0)
            opt_D.zero_grad()
            noise     = torch.randn(bs, NOISE_DIM).to(device)
            fake_imgs = G(noise, labels)
            d_loss    = (nn.BCELoss()(D(real_imgs, labels),
                         torch.ones(bs,1).to(device)*0.9) +
                         nn.BCELoss()(D(fake_imgs.detach(), labels),
                         torch.zeros(bs,1).to(device)))
            d_loss.backward(); opt_D.step()
            opt_G.zero_grad()
            noise     = torch.randn(bs, NOISE_DIM).to(device)
            fake_imgs = G(noise, labels)
            g_loss    = nn.BCELoss()(D(fake_imgs, labels),
                         torch.ones(bs,1).to(device))
            g_loss.backward(); opt_G.step()
            g_sum += g_loss.item(); d_sum += d_loss.item()
        g_avg = g_sum/len(loader); d_avg = d_sum/len(loader)
        g_losses.append(g_avg); d_losses.append(d_avg)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs} G:{g_avg:.4f} D:{d_avg:.4f}')
        if epoch % 50 == 0:
            torch.save(G.state_dict(), f'{ckpt_dir}/G_epoch_{epoch}.pth')

    torch.save(G.state_dict(), f'{ckpt_dir}/G_final_200.pth')
    with open(f'{ckpt_dir}/../losses_200.json','w') as f:
        json.dump({'g':g_losses,'d':d_losses}, f)
    print('Training complete.')

if __name__ == '__main__':
    train('dataset', 'checkpoints_200', epochs=200)
