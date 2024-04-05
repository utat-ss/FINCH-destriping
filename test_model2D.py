import torch
from torchvision import transforms, datasets, utils
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from einops import rearrange, reduce, repeat
from torchvision import transforms as T, utils
from random import random
import torch.nn.functional as F
from tqdm.auto import tqdm
from multiprocessing import cpu_count
from torch import nn, einsum
from pathlib import Path
# stdlib 
import copy 
import math
import random
import numpy as np
from PIL import Image
import spectral as spy
from functools import partial

#python3.12 -m pip install 


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def add_basic_stripes(data_cube, num_stripes=0):
    """
    Add stripes to original data set which has no stripes. Stripes are added randomly and have no specific patterns

    Args:
    data_cube: orginal data cube
    num_stripes: optional variable, if number is given, the code will generate that amount of stripes on each frame. If number not given, the code will randomly choose the number of stripes for each frame

    Returns:
    striped_data: data cube with stripes added to it
    """

    cube_dim = data_cube.shape  # cube dimesion

    striped_data = copy.deepcopy(
        data_cube
    )  # copying original data cube to not affect changes we make to it here

    if (
        num_stripes == 0
    ):  # random number of stripes for the frame, orignally 25%-75%  are striped changed to 25% to 60% of col are striped
        num_stripes = np.random.randint(
            low=int(0.25 * cube_dim[1]), high=int(0.6 * cube_dim[1]), size=cube_dim[2]
        )

    for i in range(0, cube_dim[2], 1):  # going through each frame
        col_stripes = random.sample(
            range(0, cube_dim[1]), num_stripes[i]
        )  # create a list of non repeating int numbers for size of data cube
        multiplier = (
            np.random.randint(low=5, high=16, size=num_stripes[i]) / 10
        )  # create list of repeating random number, this include number 1, we may want to prevent that

        for k in range(
            0, len(col_stripes), 1
        ):  # go through each column that we will add stripes to and do the multiplier
            striped_data[:, col_stripes[k], i] *= multiplier[k]

    return striped_data


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)


def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def divisible_by(numer, denom):
    return (numer % denom) == 0


def HSI_DS_to_IMG(HSI_datasets, bands, augmented_dataset):

    for i in range(len(HSI_datasets)):
        for j in range(bands):
            spy.save_rgb(augmented_dataset + '/HSI_' + str(i) + "_Band_" + str(j) + ".png", HSI_datasets[i], [j])
    

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def cycle(dl):
    while True:
        for data in dl:
            yield data

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


class GaussianDiffusionDestriping(GaussianDiffusion):
    def __init__(
                self,
                model,
                *,
                image_size,
                timesteps = 1000,
                sampling_timesteps = None,
                objective = 'pred_v',
                beta_schedule = 'sigmoid',
                schedule_fn_kwargs = dict(),
                ddim_sampling_eta = 0.,
                auto_normalize = True,
                offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
                min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
                min_snr_gamma = 5):
        
        super().__init__(model = model, 
                         image_size = image_size, 
                         timesteps = timesteps)


    def p_losses(self, x_start, x_stripes, t, noise = None, offset_noise_strength = None):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample
            
        #print(x_stripes.shape)
        #print(t.shape)
        #print(noise.shape)


        x = self.q_sample(x_start = x_stripes, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
    
        return loss.mean()


    def forward(self, img, img_with_stripes, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(x_start = img, x_stripes = img_with_stripes, t = t, *args, **kwargs)


    def test_destriping_sample(self, shape, striped_image, return_all_timesteps = False):
        batch, device = shape[0], self.device

        img = striped_image
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret


    def test_destriping(self, striped_image, batch_size = 16, return_all_timesteps = False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.test_destriping_sample 
        return sample_fn((batch_size, channels, image_size, image_size), striped_image, return_all_timesteps = return_all_timesteps)


class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)
    

class TrainerDestriping(Trainer):
    def __init__(
            self,
            diffusion_model,
            HSI_datset, #must be clean
            augmented_dataset_path_clean,
            *,
            train_batch_size = 110,
            gradient_accumulate_every = 1,
            augment_horizontal_flip = True,
            train_lr = 1e-4,
            train_num_steps = 100000,
            ema_update_every = 10,
            ema_decay = 0.995,
            adam_betas = (0.9, 0.99),
            save_and_sample_every = 1000,
            num_samples = 25,
            results_folder = './results',
            amp = False,
            mixed_precision_type = 'fp16',
            split_batches = True,
            convert_image_to = None,
            calculate_fid = True,
            inception_block_idx = 2048,
            max_grad_norm = 1.,
            num_fid_samples = 50000,
            save_best_and_latest_only = False):
        
        #HSI_DS_to_IMG(HSI_datset, HSI_datset.shape[-1], augmented_dataset_path_clean)

        self.training_dataset = torch.from_numpy(add_basic_stripes(HSI_datset))

        super().__init__(diffusion_model, augmented_dataset_path_clean)

        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        self.training_dataset = Dataset(augmented_dataset_path_clean, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)

        training_dataset = DataLoader(self.training_dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        training_dataset = self.accelerator.prepare(training_dataset)
        self.training_dataset = cycle(training_dataset)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    print(type(self.dl))
                    print(type(self.training_dataset))
                    data_with_stripes = next(self.training_dataset).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(img = data, img_with_stripes = data_with_stripes) 
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim = 0)

                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                        # whether to calculate fid

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')
                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')


def main():

    transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

    dataset = datasets.ImageFolder('Test_Dataset', transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    images, labels = next(iter(dataloader))

    numpy_image = images.detach().cpu().numpy()

    #print(numpy_image.shape)


    test_dataset = np.random.rand(110, 128, 128, 200)

    #dataset_with_stripes = add_basic_stripes(test_dataset)

    #print(test_dataset.shape)

    #HSI_DS_to_IMG(dataset_with_stripes, 200, "rgb_dataset")

    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        flash_attn = True
    )

    diffusion = GaussianDiffusionDestriping(
        model,
        image_size = 128,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )

    trainer = TrainerDestriping(
        diffusion,
        test_dataset,
        'rgb_dataset',
        train_batch_size = 16,
        train_lr = 8e-5,
        train_num_steps = 700000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = True              # whether to calculate fid during training
    )

    trainer.train()

if __name__ == "__main__":

    main()

