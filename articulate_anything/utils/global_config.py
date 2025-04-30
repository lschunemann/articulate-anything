# global_config.py
from articulate_anything.utils.utils import load_config

# Initialize with default config
_CONFIG = load_config()

def get_config():
    return _CONFIG

def set_config(cfg):
    global _CONFIG
    _CONFIG = cfg