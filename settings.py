import os
from dataclasses import dataclass, field    # @dataclass -> simplify class creation, field() -> customize how attributes behave
from typing import ClassVar, Dict


def get_env(key: str, default: str) -> str:
    return os.getenv(key, default)

def get_env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except ValueError:
        return default
    
@dataclass(frozen=True)
class Settings:
    # ---- Core ----
    llm_name: str = field(default_factory=lambda: get_env("LLM", "llama3"))
    