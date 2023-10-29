from base_models.simple_unet import UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sample import sample_plot_image
import numpy as np
model = UNet()
device = 'cpu'
def linear_beta_scheduler(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

IMG_SIZE = 64
BATCH_SIZE = 128
T = 100
beta = linear_beta_scheduler(T)
alpha = 1-beta
cumul_prod_alpha = torch.cumprod(alpha, dim=0)
sqrt_cumul_prod_alpha = torch.sqrt(cumul_prod_alpha)
sqrt_one_minus_cumul_prod_alpha = torch.sqrt(1-cumul_prod_alpha)

def forward_diffusion_sample(x_0, t, device="cpu"):
    noise = torch.randn_like(x_0)
    # t can be a list of timesteps (size N), this line helps us to have the corresponding alphas, its the same as torch.gather(cumul_prod_alpha, -1, t)
    sqrt_cumul_alpha = sqrt_cumul_prod_alpha[t]
    sqrt_one_minus_cumul_alpha = sqrt_one_minus_cumul_prod_alpha[t]
    # we reshape the tensor from N -> (N,1,1,1...) we add as many 1's to make these tensors the same dimension as x_0 
    # to multiply them by x_0 or the noise after (using broadcasting)
    sqrt_cumul_alpha = torch.reshape(sqrt_cumul_alpha,(t.shape[0], *((1,)*(len(x_0.shape)-1)))).to(device)
    sqrt_one_minus_cumul_alpha = torch.reshape(sqrt_one_minus_cumul_alpha,(t.shape[0], *((1,)*(len(x_0.shape)-1)))).to(device)
    # we generate the noisy image(s)
    noisy_img = sqrt_cumul_alpha.to(device)*x_0.to(device) + sqrt_one_minus_cumul_alpha.to(device)*noise.to(device), noise.to(device)
    return noisy_img


def load_transformed_dataset() : 
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)
    train = torchvision.datasets.CIFAR10(root=".", download=True,
                                         transform=data_transform)
    return train


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))

data = load_transformed_dataset()
train_data, test_data = torch.utils.data.random_split(data, [int(len(data)*0.8), len(data)-int(len(data)*0.8)])

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, drop_last=True)

def get_loss(model, x_0, t) : 
    noised_img, noise = forward_diffusion_sample(x_0, t)
    pred_noise = model(noised_img,t)
    return F.l1_loss(noise,pred_noise)


epochs = 1
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
for epoch in range(epochs):
    for step, batch in enumerate(train_dataloader):
      if step == 1000 :
        break
      optimizer.zero_grad()

      t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
      loss = get_loss(model, batch[0], t)
      loss.backward()
      optimizer.step()

      if epoch % 5 == 0 and step == 0:
        print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
        sample_plot_image(model)



## msvcrt