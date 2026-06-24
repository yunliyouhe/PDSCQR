import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class CommonConfig:
    output_dir: str
    seed: int = 20240601
    M: int = 9
    B_boot: int = 200
    eps_list: List[float] = None
    winsor_limit: float = 5.0
    export_slope_coefficients: bool = False


@dataclass
class BeijingConfig(CommonConfig):
    data_dir: str = "data/PRSA_Data_20130301-20170228"
    central_station: str = "Aotizhongxin"
    split_date: str = "2016-09-01"
    covariates: List[str] = None


@dataclass
class ACSConfig(CommonConfig):
    data_path: str = "data/acs_subset.pkl"
    n_per_state: int = 15000
    central_state: str = "Alabama"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_beijing_config(path) -> BeijingConfig:
    data = load_json(path)
    return BeijingConfig(**data)


def load_acs_config(path) -> ACSConfig:
    data = load_json(path)
    return ACSConfig(**data)


def resolve_path(path: str, repo_root: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return repo_root / p
