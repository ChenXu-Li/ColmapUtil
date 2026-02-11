"""
配置文件加载器
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


_config: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，如果为None，则使用默认路径
        
    Returns:
        配置字典
    """
    global _config
    
    if config_path is None:
        # 使用项目根目录下的config.yaml
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        _config = yaml.safe_load(f)
    
    return _config


def get_config(key: Optional[str] = None, default: Any = None) -> Any:
    """
    获取配置值
    
    Args:
        key: 配置键，支持点号分隔的嵌套键（如 'paths.input_base'）
        default: 默认值
        
    Returns:
        配置值
    """
    global _config
    
    if _config is None:
        load_config()
    
    if key is None:
        return _config
    
    # 支持嵌套键访问
    keys = key.split('.')
    value = _config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    
    return value
