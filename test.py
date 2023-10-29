from train.train import Trainer
from diffusion_models.gaussian_diffusion import GaussianDiffusion
from base_models.complex_unet import Unet

unet = Unet(64)
image_size = 128
diffusion_model = GaussianDiffusion(unet, image_size=image_size, timesteps=100, sampling_timesteps=10)

trainer = Trainer(diffusion_model, 'images/', train_num_steps=1, calculate_fid=False)


if __name__ == '__main__':
    trainer.train()
#..\..\venv\cuda_pytorch\Scripts\Activate.ps1