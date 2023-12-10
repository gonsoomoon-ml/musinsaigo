from enum import Enum


class DirName(str, Enum):
    CONFIGS: str = "configs"
    LOGS: str = "logs"


class FileName(str, Enum):
    CONFIG: str = "config.yaml"
    LOG: str = "log.txt"
