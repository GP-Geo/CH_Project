"""Training layer for channel-heads models.

Houses the model *training* code, kept separate from the model *definitions*
(:mod:`channel_heads.models`). Currently provides the shared OutletCNN training
core in :mod:`channel_heads.training.cnn` (training loop + hyperparameter
defaults). Requires PyTorch.
"""
