# phasen_remixit

## Training
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --configuration config/train_RATS/phasen_vb.json --preloaded_model_path ../speech_enhance_phasen/Experiments/phasen_vb/checkpoints/best_model.tar
```
## Inference

```bash
source run_inference.sh
```

## Note

So far not working, for different SNR
- fixed or EMA
- lower lr
- student pre-trained or not

TBC
- add SI-SDR in loss function
