{

PYTHONPATH='.':$PYTHONPATH python synthetic/train_generative_model.py \
    --real_alg_name BetaVAE \
    --name dsprites_full_BetaVAE \
    --dset_name dsprites_full

    exit
}
