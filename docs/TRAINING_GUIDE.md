# Training guide

`scripts/train.py --config configs/models/<model>.yaml` dispatches to actual training. State models use AdamW, cosine scheduler, gradient accumulation/clipping, train/validation, early stopping, nonfinite detection, checkpoint and RNG resume. Visual, video and joint paths use AdamW, scheduler, clipping and checkpoints; validation, resume and mixed precision remain pending there. Pilot runs used FP32 CPU because the installed PyTorch wheel has no CUDA support.

Run directories must be new. `last.pt` includes weights, optimizer, scheduler and RNG. Visual checkpoints save resume data but do not yet implement resume. Consult `metrics.jsonl` and state-model `manifest.json` for raw output.
