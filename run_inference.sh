#!/bin/bash



CUDA_VISIBLE_DEVICES=2 python inference.py -mc Experiments/phasen_vb/config.json \
    -dc config/inference/test_vb.json \
    -cp Experiments/phasen_vb/checkpoints/best_model.tar \
    -dist output/phasen_remixit_awgn_snr_10