from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler


def default_voc_scheduler(epochs=12, decay_epochs=9, warmup_epochs=0):
    """
    VOC schedule based on batch size of 16.
    VOC 07+12 trainval has ~16551 images.
    1 epoch ~= 1035 iterations.
    """
    # total number of iterations assuming 16 batch size, using 16551/16=1035
    iters_per_epoch = 1050
    total_steps = epochs * iters_per_epoch
    warmup_steps = warmup_epochs * iters_per_epoch

    if isinstance(decay_epochs, (list, tuple)):
        milestones = [x * iters_per_epoch for x in decay_epochs]
        values = [1.0]
        for _ in range(len(decay_epochs)):
            values.append(values[-1] * 0.1)
    else:
        milestones = [decay_epochs * iters_per_epoch]
        values = [1.0, 0.1]

    scheduler = L(MultiStepParamScheduler)(
        values=values,
        milestones=milestones + [total_steps],
    )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / total_steps,
        warmup_method="linear",
        warmup_factor=0.001,
    )


# Default schedule: 12 epochs total, decay at 9 epochs
lr_multiplier_12ep = default_voc_scheduler(epochs=12, decay_epochs=9, warmup_epochs=0)
lr_multiplier_50ep = default_voc_scheduler(epochs=50, decay_epochs=40, warmup_epochs=0)
