{

PYTHONPATH='.':$PYTHONPATH python synthetic/discover_unknown_bias.py \
    --seed 123 \
    --dset_name dsprites_full \
    --real_alg_name BetaVAE \
    --num_workers 0 \
    --latent_vec_dir ckpt/synthetic/latent_code \
    --include_labels 0 1 2 3 4 5 \
    --lr 1e-3 \
    --biased_normal_vec_save_dir ckpt/synthetic/discovered_biased_hyperplane \
    --ckpt_load ckpt/synthetic/generative_model/dsprites_full_BetaVAE/last \
    --lambda_orthogonal_constraint 10 \
    --name dsprites_full_BetaVAE_w_orth_b2t1 \
    --bias_attr_index 2 \
    --target_attr_index 1 \
    --biased_classifier_ckpt ckpt/synthetic/classifier/dsprites_full_target_1_bias_2.pth \
    --normal_vector_npz_fpath ckpt/synthetic/gt_hyperplanes/BetaVAE_dsprites_full.npz

    exit
}
