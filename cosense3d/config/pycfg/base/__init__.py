from cosense3d.utils.train_utils import get_gpu_architecture


gpu_arc = get_gpu_architecture()
if gpu_arc >= 75:
    use_flash_attn = True
    attn = 'MultiheadFlashAttention'
else:
    use_flash_attn = True
    attn = 'MultiheadAttention'