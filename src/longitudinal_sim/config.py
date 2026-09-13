from dataclasses import asdict, dataclass
from dataclasses import field


@dataclass
class ExperimentConfig:
    n_subjects: int = 300
    max_visits: int = 12
    min_visits: int = 3
    n_timevarying: int = 3
    n_static: int = 4
    n_classes: int = 3
    hidden_size: int = 32
    rnn_type: str = "gru"
    batch_size: int = 32
    pretrain_epochs: int = 5
    finetune_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    mask_rate: float = 0.35
    self_supervised_mask_rate: float = 0.15
    missingness: str = "label_dependent"
    dgp_profile: str = "default"
    info_profile: str = "balanced"
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    seed: int = 7
    lambda_rec: float = 1.0
    lambda_cons: float = 0.2
    lambda_task: float = 1.0
    lambda_clin: float = 0.3
    lambda_sum: float = 0.4
    lambda_align: float = 0.15
    lambda_motif: float = 0.45
    lambda_image_motif: float = 0.08
    image_size: int = 16
    downstream_models: list[str] = field(
        default_factory=lambda: [
            "mixed_effects_ordlogit",
            "longitudinal_rf",
        ]
    )
    imputation_methods: list[str] = field(
        default_factory=lambda: ["brits_joint", "brits_impute_only", "mean", "forward_fill", "iterative", "r_mice", "r_missforest"]
    )

    def to_dict(self) -> dict:
        return asdict(self)
