from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    symbol: str = "AAPL"
    start: str = "2020-01-01"
    end: str = "2025-01-01"
    window_size: int = 30
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    batch_size: int = 32
    epochs: int = 60
    learning_rate: float = 1e-3
    lstm_units: int = 64
    dropout_rate: float = 0.2
    random_seed: int = 42
    artifacts_root: Path = Path("artifacts")

    @property
    def artifact_dir(self) -> Path:
        return self.artifacts_root / self.symbol.upper()
