##   ET-Talk: Effective Training Strategy to Enhance Fidelity and Sychrony for Talking Face Generation (ICME 2025)

Implementation of ET-Talk


### Requirements

Use conda to create an environment and prepare with following command

> pip install -r requirment.txt

I have tested on:

- PyTorch 2.4.1
- CUDA 12.4 / CUDA 11.8

### Inference

First download the pretrain checkpoint and then run the inference code:

> python styleSync_inference.py --face VIDEO_PATH --audio AUDIO_PATH --ckpt CKPT_PATH --outfile OUT_PATH

The results will be saved in OUT_PATH.

### Pretrained Checkpoints

[Link]()


### License

