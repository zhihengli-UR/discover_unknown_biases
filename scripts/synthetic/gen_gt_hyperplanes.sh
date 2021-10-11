{

PYTHONPATH=.:$PYTHONPATH python synthetic/gen_gt_hyperplanes.py \
    --data_root ckpt/synthetic/latent_code \
    --real_alg_name BetaVAE \
    --dset_name dsprites_full \
    --output_dir ckpt/synthetic/gt_hyperplanes \
    --epoch 500

    exit
}
