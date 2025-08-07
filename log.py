import psutil
import torch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_system_info():
    """Log system resource information"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        gpu_info = []
        if torch.cuda.is_available():
            for gpu in GPUtil.getGPUs():
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': f"{gpu.load*100}%",
                    'memory_used': f"{gpu.memoryUsed}MB/{gpu.memoryTotal}MB",
                    'temperature': f"{gpu.temperature}°C"
                })
        logger.info(f"System Info - CPU: {cpu_percent}%, RAM: {memory.percent}%, "
                   f"Available RAM: {memory.available/1024/1024/1024:.1f}GB")
        if gpu_info:
            logger.info(f"GPU Info: {gpu_info}")
    except Exception as e:
        logger.warning(f"Failed to log system info: {str(e)}")