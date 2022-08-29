# Denoising_Diffusion_Implicit_Models
A replication of Denoising Diffusion Implicit Models paper with PyTorch and [ViT](https://github.com/facebookresearch/dino).

![img](https://github.com/EBGU/Denoising_Diffusion_Implicit_Models/blob/main/Saved_Models/samples_flower.png)

![img](https://github.com/EBGU/Denoising_Diffusion_Implicit_Models/blob/main/Saved_Models/denoise_sequence_flower.png)

To train a new model, you can modify the yaml file and:

` python multi_gpu_trainer.py 20220822 `

Or you can download my pretrained weights for [Oxford Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/) and run inference:

` python ViT.py `

The inference process is controled by 3 parameters (DiffusionVisionTransformer.sampler):

"device", usually `torch.device('cuda')` ;

"k", because the sampling become deterministic with [DDIM](https://arxiv.org/abs/2010.02502), we can jump k steps for faster speed;

"N", how many samples you will get.

The result should looks like the welcoming images.

If you want my pretrained weights for miniImageNet, please leave you email address. Enjoy!

### 20220829
Upload checkpoints for high-resolution images(200*200), and add new code for [zero-shot draft-to-drawings](https://github.com/EBGU/Denoising_Diffusion_Implicit_Models/blob/main/ViT_draft2drawing.py).
![img](https://github.com/EBGU/Denoising_Diffusion_Implicit_Models/blob/main/Saved_Models/d2d_sample.jpg)
