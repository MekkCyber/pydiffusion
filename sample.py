import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import numpy as np
def linear_beta_scheduler(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

IMG_SIZE = 64
BATCH_SIZE = 128
T = 100
beta = linear_beta_scheduler(T)
alpha = 1-beta
sqrt_inverse_alphas = torch.sqrt(1.0 / alpha)
cumul_prod_alpha = torch.cumprod(alpha, dim=0)
sqrt_cumul_prod_alpha = torch.sqrt(cumul_prod_alpha)
sqrt_one_minus_cumul_prod_alpha = torch.sqrt(1-cumul_prod_alpha)

alphas_cumprod_prev = F.pad(cumul_prod_alpha[:-1], (1, 0), value=1.0)
posterior_variance = beta * (1. - alphas_cumprod_prev) / (1. - cumul_prod_alpha)

@torch.no_grad()
def sample_inference(x, t, model, device='cpu') :
    beta_t = beta[t]
    beta_t = torch.reshape(beta_t,(t.shape[0], *((1,)*(len(x.shape)-1)))).to(device)
    sqrt_one_minus_cumul_prod_alpha_t = sqrt_one_minus_cumul_prod_alpha[t]
    sqrt_one_minus_cumul_prod_alpha_t = torch.reshape(sqrt_one_minus_cumul_prod_alpha_t,(t.shape[0], *((1,)*(len(x.shape)-1)))).to(device)
    sqrt_alpha_inverse_t = sqrt_inverse_alphas[t]
    sqrt_alpha_inverse_t = torch.reshape(sqrt_alpha_inverse_t,(t.shape[0], *((1,)*(len(x.shape)-1)))).to(device)

    mean = sqrt_alpha_inverse_t*(x-(beta_t)/(sqrt_one_minus_cumul_prod_alpha_t)*model(x,t))
    posterior_variance_t = posterior_variance[t]
    posterior_variance_t = torch.reshape(posterior_variance_t,(t.shape[0], *((1,)*(len(x.shape)-1)))).to(device)
    if t==0 : 
        return mean
    else : 
        z = torch.rand_like(x)
        return mean + posterior_variance_t*z

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))

@torch.no_grad()
def sample_plot_image(model, device='cpu'):
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_inference(img, t, model)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu())
    plt.show()


