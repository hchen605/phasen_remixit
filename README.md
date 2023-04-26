# phasen_remixit

## Training
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --configuration config/train_RATS/phasen_vb.json --preloaded_model_path ../speech_enhance_phasen/Experiments/phasen_vb/checkpoints/best_model.tar
```
## Inference

```bash
source run_inference.sh
```