import tomllib
from dataclasses import dataclass
from data.driver_and_solution_info import RDE, Driver
from builtins import type
from typing import Literal
from abc import ABC, abstractmethod


@dataclass
class DatasetConfig:
    driver: Driver
    rde: RDE
    hurst: float
    signature_type: Literal["signature", "log_signature"]
    signature_depth: int
    num_steps: int
    num_paths: int
    dim: int

    train_percentage: float = 0.8
    val_percentage: float = 0.1
    test_percentage: float = 0.1

    def __post_init__(self):
        if self.driver not in Driver:
            raise ValueError(f"Driver {self.driver} not supported. Choose from {Driver}")
        if self.rde not in RDE:
            raise ValueError(f"RDE {self.rde} not supported. Choose from {RDE}")
        if self.signature_type not in ["signature", "log_signature"]:
            raise ValueError(f"Signature type {self.signature_type} not supported. Choose from ['signature', 'log_signature']")
        if self.signature_depth <= 0 or self.signature_depth > 5:
            raise ValueError(f"Signature depth must be positive and less than 5. Got {self.signature_depth}")
        if self.num_steps <= 0:
            raise ValueError(f"Number of steps must be positive. Got {self.num_steps}")
        if self.num_paths <= 0:
            raise ValueError(f"Number of paths must be positive. Got {self.num_paths}")


@dataclass
class RDEConfig(ABC):
    @abstractmethod
    def __post_init__(self):
        pass


@dataclass
class OUConfig(RDEConfig):
    lam: float
    sigma: float
    y0: float

    def __post_init__(self):
        if self.lam <= 0:
            raise ValueError(f"Lambda must be positive. Got {self.lam}")
        if self.sigma <= 0:
            raise ValueError(f"Sigma must be positive. Got {self.sigma}")
        if self.y0 <= 0:
            raise ValueError(f"Y0 must be positive. Got {self.y0}")


@dataclass
class DataConfig:
    epochs: int
    batch_size: int
    seed: int

    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError(f"Epochs must be positive. Got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive. Got {self.batch_size}")
        if self.seed <= 0:
            raise ValueError(f"Seed must be positive. Got {self.seed}")


@dataclass
class ModelConfig:
    use_rough_paths: bool
    hidden_dim: int
    num_resno_blocks: int

    def __post_init__(self):
        if self.use_rough_paths not in [True, False]:
            raise ValueError(f"Use rough paths must be a boolean. Got {self.use_rough_paths}")
        if self.hidden_dim <= 0:
            raise ValueError(f"Hidden dimension must be positive. Got {self.hidden_dim}")
        if self.num_resno_blocks <= 0:
            raise ValueError(f"Number of RESNO blocks must be positive. Got {self.num_resno_blocks}")


@dataclass
class AutoencoderConfig:
    C_AE: float
    C_e: float


@dataclass
class ExperimentConfig:
    dataset_config: DatasetConfig
    rde_config: RDEConfig
    data_config: DataConfig
    model_config: ModelConfig
    autoencoder_config: AutoencoderConfig

    @classmethod
    def from_toml(cls: type["ExperimentConfig"], path: str) -> "ExperimentConfig":
        with open(path, "rb") as f:
            config = tomllib.load(f)

        dataset_config = DatasetConfig(
            driver=config["data_config"]["driver"],
            rde=config["data_config"]["rde"],
            hurst=config["data_config"]["hurst"],
            signature_type=config["data_config"]["signature_type"],
            signature_depth=config["data_config"]["signature_depth"],
            num_steps=config["data_config"]["num_steps"],
            num_paths=config["data_config"]["num_paths"],
            dim=config["data_config"]["dim"],
            train_percentage=config["data_config"]["train_percentage"],
            val_percentage=config["data_config"]["val_percentage"],
            test_percentage=config["data_config"]["test_percentage"],
        )

        match config["data_config"]["rde"]:
            case "fOU":
                rde_config = OUConfig(
                    lam=config["data_config"]["lam"],
                    sigma=config["data_config"]["sigma"],
                    y0=config["data_config"]["y0"],
                )
            case _:
                raise ValueError(f"RDE {config['data_config']['rde']} not supported. Choose from {'fOU'}")

        data_config = DataConfig(
            epochs=config["data_config"]["epochs"],
            batch_size=config["data_config"]["batch_size"],
            seed=config["data_config"]["seed"],
        )

        model_config = ModelConfig(
            use_rough_paths=config["model_config"]["use_rough_paths"],
            hidden_dim=config["model_config"]["hidden_dim"],
            num_resno_blocks=config["model_config"]["num_resno_blocks"],
        )

        autoencoder_config = AutoencoderConfig(
            C_AE=config["autoencoder_config"]["C_AE"],
            C_e=config["autoencoder_config"]["C_e"],
        )

        return cls(
            dataset_config=dataset_config,
            rde_config=rde_config,
            data_config=data_config,
            model_config=model_config,
            autoencoder_config=autoencoder_config,
        )


if __name__ == "__main__":
    print(ExperimentConfig.from_toml("resno/configs/config.toml"))
