{

PYTHONPATH='.':$PYTHONPATH python synthetic/gen_latent_code.py \
    --real_alg_name BetaVAE \
    --name dsprites_full_BetaVAE \
    --latent_vec_dir ckpt/synthetic/latent_code \
    --ckpt_load ckpt/synthetic/generative_model/dsprites_full_BetaVAE/last \
    --test true \
    --include_labels 0 1 2 3 4 5 \
    --dset_name dsprites_full

    exit
}