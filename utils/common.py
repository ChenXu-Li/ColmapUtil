"""
公共工具函数
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from .config_loader import get_config


def setup_environment():
    """
    设置环境变量（避免OpenBLAS段错误等）
    """
    env_vars = get_config('environment', {})
    for key, value in env_vars.items():
        os.environ[key] = str(value)


def check_dependencies():
    """
    检查必要的依赖是否已安装
    """
    missing = []
    
    try:
        import pycolmap
    except ImportError:
        missing.append('pycolmap')
    
    try:
        import numpy
    except ImportError:
        missing.append('numpy')
    
    try:
        import cv2
    except ImportError:
        missing.append('opencv-python')
    
    try:
        import viser
    except ImportError:
        missing.append('viser')
    
    if missing:
        print(f"❌ 缺少以下依赖: {', '.join(missing)}")
        print(f"请运行: pip install {' '.join(missing)}")
        sys.exit(1)
    
    print("✅ 所有依赖已安装")


def get_project_root() -> Path:
    """
    获取项目根目录
    """
    return Path(__file__).parent.parent


def check_port(port: int) -> bool:
    """
    检查端口是否可用
    
    Args:
        port: 端口号
        
    Returns:
        True表示端口已被占用，False表示可用
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('0.0.0.0', port))
    sock.close()
    return result == 0


def format_duration(seconds: int) -> str:
    """
    格式化时间显示
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}小时 {minutes}分钟 {secs}秒"
    elif minutes > 0:
        return f"{minutes}分钟 {secs}秒"
    else:
        return f"{secs}秒"
