Note on mixed precision fine-tuning on NVIDIA GPUs
Full precision is recommended for fine-tuning. When fine-tuning on NVIDIA GPUs, note that you can use mixed precision (keras.mixed_precision.set_global_policy('mixed_bfloat16')) to speed up training with minimal effect on training quality. Mixed precision fine-tuning does consume more memory so is useful only on larger GPUs.

For inference, half-precision (keras.config.set_floatx("bfloat16")) will work and save memory while mixed precision is not applicable.