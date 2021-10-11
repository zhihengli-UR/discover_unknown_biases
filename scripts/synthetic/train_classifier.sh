{

PYTHONPATH=.:$PYTHONPATH python synthetic/train_classifier.py \
  --dset_name dsprites_full \
  --bias_attr_index 2 \
  --target_attr_index 1 \
  --num_workers 0

    exit
}