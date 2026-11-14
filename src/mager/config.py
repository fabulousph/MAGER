"""Configuration for MAGER."""
import os
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"


class ModelConfig:
    def __init__(self, config_dict: Dict):
        self.api_type = config_dict.get("api_type", "openai")
        self.base_url = config_dict.get("base_url")
        self.api_key = config_dict.get("api_key", "EMPTY")
        self.model_name = config_dict.get("model_name", "qwen3-8b")
        self.temperature = config_dict.get("temperature", 0.1)
        self.max_tokens = config_dict.get("max_tokens", 3000)
        self.timeout = config_dict.get("timeout", 120)
        self.headers = config_dict.get("headers", {})


CONFIG = {
    "agent_a": {
        "api_type": os.getenv("AGENT_A_API_TYPE", "openai"),
        "base_url": os.getenv("AGENT_A_BASE_URL", "http://127.0.0.1:8000/v1"),
        "api_key": os.getenv("AGENT_A_API_KEY", "EMPTY"),
        "model_name": os.getenv("AGENT_A_MODEL", "qwen3-8b"),
        "temperature": float(os.getenv("AGENT_A_TEMPERATURE", "0.3")),
        "max_tokens": int(os.getenv("AGENT_A_MAX_TOKENS", "3000")),
        "timeout": int(os.getenv("AGENT_A_TIMEOUT", "120")),
    },
    "agent_g": {
        "api_type": os.getenv("AGENT_G_API_TYPE", "openai"),
        "base_url": os.getenv("AGENT_G_BASE_URL", "http://127.0.0.1:8000/v1"),
        "api_key": os.getenv("AGENT_G_API_KEY", "EMPTY"),
        "model_name": os.getenv("AGENT_G_MODEL", "qwen3-8b"),
        "temperature": float(os.getenv("AGENT_G_TEMPERATURE", "0.8")),
        "max_tokens": int(os.getenv("AGENT_G_MAX_TOKENS", "3000")),
        "timeout": int(os.getenv("AGENT_G_TIMEOUT", "120")),
    },
    "agent_d": {
        "api_type": os.getenv("AGENT_D_API_TYPE", "openai"),
        "base_url": os.getenv("AGENT_D_BASE_URL", "http://127.0.0.1:8000/v1"),
        "api_key": os.getenv("AGENT_D_API_KEY", "EMPTY"),
        "model_name": os.getenv("AGENT_D_MODEL", "qwen3-8b"),
        "temperature": float(os.getenv("AGENT_D_TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("AGENT_D_MAX_TOKENS", "3000")),
        "timeout": int(os.getenv("AGENT_D_TIMEOUT", "120")),
    },
    "agent_e": {
        "api_type": os.getenv("AGENT_E_API_TYPE", "openai"),
        "base_url": os.getenv("AGENT_E_BASE_URL", "http://127.0.0.1:8000/v1"),
        "api_key": os.getenv("AGENT_E_API_KEY", "EMPTY"),
        "model_name": os.getenv("AGENT_E_MODEL", "qwen3-8b"),
        "temperature": float(os.getenv("AGENT_E_TEMPERATURE", "0.2")),
        "max_tokens": int(os.getenv("AGENT_E_MAX_TOKENS", "3000")),
        "timeout": int(os.getenv("AGENT_E_TIMEOUT", "120")),
    },
}


DATASETS = {
    "politifact": {
        "misinformation": DATA_ROOT / "politifact" / "politifact_fake.converted.jsonl",
        "verified_information": DATA_ROOT / "politifact" / "politifact_real.converted.jsonl",
        "user_instances": DATA_ROOT / "politifact" / "users.converted.jsonl",
        "enhanced": DATA_ROOT / "politifact" / "enhanced_dataset_politifact_qwen3-8b.jsonl",
    },
    "gossipcop": {
        "misinformation": DATA_ROOT / "gossipcop" / "gossipcop_fake.converted.jsonl",
        "verified_information": DATA_ROOT / "gossipcop" / "gossipcop_real.converted.jsonl",
        "user_instances": DATA_ROOT / "gossipcop" / "users.converted.jsonl",
        "enhanced": DATA_ROOT / "gossipcop" / "enhanced_dataset_gossipcop_qwen3-8b.jsonl",
    },
    "mcfend": {
        "misinformation": DATA_ROOT / "mcfend" / "news.false.jsonl",
        "verified_information": DATA_ROOT / "mcfend" / "news.true.jsonl",
        "user_instances": DATA_ROOT / "mcfend" / "users.converted.jsonl",
        "enhanced": DATA_ROOT / "mcfend" / "enhanced_dataset_mcfend_qwen3-8b.jsonl",
    },
}


def set_dataset(dataset_name: str):
    global DATA_PATHS
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASETS)}")
    DATA_PATHS = {key: str(value) for key, value in DATASETS[dataset_name].items()}
    os.environ["MAGER_DATASET"] = dataset_name


set_dataset(os.getenv("MAGER_DATASET", "politifact"))


GA_CONFIG = {
    "population_size": int(os.getenv("MAGER_POPULATION_SIZE", "20")),
    "max_iterations": int(os.getenv("MAGER_MAX_ITERATIONS", "100")),
    "tournament_size": 3,
    "crossover_prob": 0.6,
    "mutation_prob": 0.4,
    "top_k_retrieval": 4,
    "lambda_structure": 0.5,
    "epsilon": 1e-5,
    # 推理质量权重 μ：式(7) R(M) = R_det(M) + μ * R_rea(M)
    "mu_reasoning": float(os.getenv("MAGER_MU_REASONING", "0.25")),
    "max_workers": int(os.getenv("MAGER_MAX_WORKERS", "20")),
}

ENHANCEMENT_CONFIG = {
    "user_batch_size": 100,
    "edge_batch_size": 100,
    "max_workers": 20,
    "parallel_batch_threshold": 3,
    "batch_parallel_workers": 5,
}

ENCODER_CONFIG = {
    "model_name": os.getenv("MAGER_ENCODER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    "local_model_path": os.getenv("MAGER_ENCODER_PATH", ""),
    "device": os.getenv("MAGER_ENCODER_DEVICE", "cpu"),
    "batch_size": 32,
}

LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "mager.log",
}


def get_model_config(agent_name: str) -> ModelConfig:
    if agent_name not in CONFIG:
        raise ValueError(f"Unknown agent name: {agent_name}")
    return ModelConfig(CONFIG[agent_name])


def update_config_from_file(config_file: str):
    import json
    import yaml

    with open(config_file, "r", encoding="utf-8") as f:
        custom_config = yaml.safe_load(f) if config_file.endswith((".yaml", ".yml")) else json.load(f)
    CONFIG.update(custom_config.get("agents", {}))
    GA_CONFIG.update(custom_config.get("ga_config", {}))
