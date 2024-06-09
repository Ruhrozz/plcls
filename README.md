# Pytorch Lighning Classification 

## Description

- Train any `torchvision` model on any `torchvision` dataset.
- Use Pytorch Lighning to reduce code size.
- Create `Callbacks` to split every part of Deep Learning models and methods.

## How to use

### Prepare environment

Install `python >= 3.9`:

```bash
conda create -n plcls python=3.9 -y
conda deactivate && conda activate plcls
```

### Start training

Yaml config is all you need:

```bash
python main.py configs/config.yaml
```

### Get results

See `lightning_logs/version_0/` folder for training results.

See `lightning_logs/version_0/tb/` for tensorboard plots:

```bash
tensorboard --logdir lightning_logs/
```

## Roadmap

- [x] Training code
- [ ] Evaluation
- [ ] Custom datasets
- [ ] Custom models
- [ ] SAM, more augs, etc. (efficient training)
- [ ] Tests, linters, pipelines

