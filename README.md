# Score-based Diffusion Models in Function Space

This is a codebase for the following paper

**Score-based Diffusion Models in Function Space**

by Jae Hyun Lim\*, Nikola B Kovachki\*, Ricardo Baptista\*, Christopher Beckham, Kamyar Azizzadenesheli, Jean Kossaifi, Vikram Voleti, Jiaming Song, Karsten Kreis, Jan Kautz, Christopher Pal, Arash Vahdat, Anima Anandkumar

[[arXiv](https://arxiv.org/abs/2302.07400)]

## Experiments on Gaussian Mixture, Navier-Stokes, and Volcano Dataset
We will update the repo soon.

## Experiments on MNIST-SDF
Here's example command lines for training DDO and GANO models

### DDO
```
python main.py --command_type=train \
  --exp_path=${EXP_PATH} \
  --seed=1 --print_every=1000 --save_every=5000 --ckpt_every=100000 --eval_every=50000 --vis_every=10000 --resume \
  --data=${DATA_PATH} --dataset=mnistsdf_32 --train_img_height=32 --input_dim=1 --coord_dim=2 --transform=sdf \
  --model=fnounet2d --use_pos --modes=32 --ch=64 --ch_mult=1,2,2 --num_res_blocks=4 --dropout=0.0 --norm=group_norm --use_pointwise_op \
  --ns_method=vp_cosine --timestep_sampler=low_discrepancy \
  --disp_method=sine --sigma_blur_min=0.05 --sigma_blur_max=0.25 \
  --gp_type=exponential --gp_exponent=2.0 --gp_length_scale=0.05 --gp_sigma=1.0 \
  --num_steps=250 --sampler=denoise --s_min=0.0001 \
  --train_batch_size=32 --lr=0.0001 --weight_decay=0.0 --num_iterations=2000000 \
  --upsample --upsample_resolution=64 \
  --eval_use_ema --ema_decay=0.999 --eval_img_height=64 --eval_batch_size=256 --eval_num_samples=5000 --eval_resize_mode=tensor --eval_interpolation=bilinear --fid_dir={FID_PATH} \
  --vis_batch_size=36
```


### GANO

```
python gano.py --command_type train \
  --exp_path=${EXP_PATH} \
  --seed=1 --print_every=1000 --save_every=5000 --ckpt_every=100000 --eval_every=20000 --vis_every=10000 --resume \
  --data=${DATA_PATH} --dataset=mnistsdf_32 --train_img_height=32 --input_dim=1 --coord_dim=2 \
  --model=gano-uno --modes=32 --d_co_domain=64 --lmbd_grad=10.0 --n_critic=10 \
  --train_batch_size=32 --lr=0.0001 --weight_decay=0.0 --num_iterations=1000000 \
  --upsample --upsample_resolution=64 \
  --eval_fid --eval_use_ema --ema_decay=0.999 --eval_img_height=64 --eval_batch_size=512 --eval_num_samples=50000 --eval_resize_mode=tensor --eval_interpolation=bilinear --fid_dir={FID_PATH} \
  --vis_batch_size=36
```

### Evaluations
Run following notebook files
- `notebooks/mnistsdf_sample_ddo.ipynb`
- `notebooks/mnistsdf_sample_gano.ipynb`
  
### Pre-trained models (Google Drive)
- DDO  [[link](https://drive.google.com/file/d/1aMKEIEMI2sZKeK0TbFwHxDUNh6B-bP2l/view)]
- GANO [[link](https://drive.google.com/file/d/1aDa6sf5WFbW85kiTewvJbN55fhFZbx1M/view)]

## License
MIT License

## Citation
```
@article{lim2023score,
  title={Score-based diffusion models in function space},
  author={Lim\*, Jae Hyun and Kovachki\*, Nikola B and Baptista\*, Ricardo and Beckham, Christopher and Azizzadenesheli, Kamyar and Kossaifi, Jean and Voleti, Vikram and Song, Jiaming and Kreis, Karsten and Kautz, Jan and others},
  journal={arXiv preprint arXiv:2302.07400},
  year={2023}
}
```
