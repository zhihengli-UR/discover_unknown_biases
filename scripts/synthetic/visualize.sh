{

PYTHONPATH='.':$PYTHONPATH python synthetic/visualize.py \
    --name dsprites_full_BetaVAE_b2t1 \
    --ckpt_load ckpt/synthetic/generative_model/dsprites_full_BetaVAE/last \
    --dset_dir=data \
    --dset_name=dsprites_full \
    --traverse_z=false \
    --num_workers 0 \
    --seed 123 \
    --vis_root ckpt/synthetic/visualization \
    --bias_attr_index 2 \
    --target_attr_index 1 \
    --pred_bias_hyperplane_fpath ckpt/synthetic/discovered_biased_hyperplane/dsprites_full_BetaVAE_w_orth_b2t1.npz \
    --start_distance -5 \
    --end_distance 5 \
    --steps 6 \
    --real_alg_name=BetaVAE

    exit
}