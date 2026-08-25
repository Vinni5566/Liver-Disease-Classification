"""Central configuration shared by baseline and ROI experiments."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    model_name: str = "resnet50"  # resnet50, efficientnet_b0, or efficientnet_b3
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-5
    patience: int = 10
    freeze_backbone: bool = False
    freeze_batchnorm: bool = False
    imbalance_strategy: str = "sampler"  # sampler, weighted_loss, or both
    use_roi_annotations: bool = False
    roi_mode: str = "crop"  # crop or mask
    confidence_threshold: float = 0.70
    seed: int = 42
    experiment_name: Optional[str] = None
    resume_checkpoint: Optional[str] = None
