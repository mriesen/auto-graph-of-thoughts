from typing import Callable

N_VEC_ENVS = 8

POLICY_KWARGS = dict(
    net_arch=dict(pi=[64, 64], vf=[64, 64])
)

def clip_range_schedule(cr_start, cr_end) -> Callable[[float], float]:
    return lambda progress: cr_end + progress * (cr_start - cr_end)

CLIP_RANGE = clip_range_schedule(0.15, 0.3)
ENT_COEF = 0.03
N_EPOCHS = 8

def lr_schedule(lr_start, lr_end, warmup_fraction) -> Callable[[float], float]:
    return lambda progress: (
        lr_end + ((1.0 - progress) / warmup_fraction) * (lr_start - lr_end)
        if (1.0 - progress) < warmup_fraction
        else lr_end + (1.0 - (1.0 - progress - warmup_fraction) / (1.0 - warmup_fraction)) * (lr_start - lr_end)
    )


LEARNING_RATE = lr_schedule(5e-4, 1e-6, 0.1)
