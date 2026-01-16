"""
:Description: Controly actualización de los parámetros generales de ejecución
:Author:
    - Ana María García Serrano
    - Enrique Garcia Arias
:Organization: UNED
"""
from sources.common.common import processControl
from sources.common.utils import configLoader, dbTimestamp
import argparse
import os
import sys
import socket
import torch
import psutil

# Constants for parameter files
JSON_PARMS = "config.json"

def get_system_memory_info():
    """Obtiene información de memoria del sistema"""
    info = {
        "vram_available": 0,
        "ram_available": 0,
        "gpu_name": None,
        "has_cuda": False
    }

    # VRAM de GPU
    if torch.cuda.is_available():
        info["has_cuda"] = True
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_available"] = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        info["vram_free"] = torch.cuda.get_device_properties(0).total_memory / 1e9 - torch.cuda.memory_allocated() / 1e9

    # RAM del sistema
    ram = psutil.virtual_memory()
    info["ram_available"] = ram.available / 1e9  # GB
    info["ram_total"] = ram.total / 1e9

    return info


def select_configuration_based_on_memory():
    """
    Selecciona configuración óptima basada en memoria disponible
    Retorna: (config_name, config_dict)
    """
    mem_info = get_system_memory_info()

    if not mem_info["has_cuda"]:
        # Solo CPU - configuración mínima
        config = "cpu_only"
        return config, processControl.defaults.get('modelConfig', {}).get(config, {})

    vram_gb = mem_info["vram_available"]

    # Configuraciones por nivel de VRAM
    if vram_gb < 9:
        return "low_vram_4bit", {
            "name": "low_vram_4bit",
            "description": "VRAM < 4GB - Cuantización 4-bit extrema",
            "vram_threshold": 4,
            "use_cuda": True,
            "load_in_4bit": True,
            "load_in_8bit": False,
            "torch_dtype": torch.float16,
            "num_beams": 1,
            "max_new_tokens": 128,
            "batch_size": 1,
            "temperature": 0.2,
            "top_p": None,
            "top_k": None,
            "repetition_penalty": 1.0,
            "use_flash_attn": False,
            "enable_cpu_offload": True,
            "max_memory": {0: "8GB", "cpu": "8GB"}
        }

    elif vram_gb < 9:
        config = "medium_vram_4bit"
        return config, processControl.defaults.get('modelConfig', {}).get(config, {})

    elif vram_gb < 12:
        return "high_vram_8bit", {
            "name": "high_vram_8bit",
            "description": "VRAM 8-12GB - Cuantización 8-bit balanceada",
            "vram_threshold": 12,
            "use_cuda": True,
            "load_in_4bit": True,
            "load_in_8bit": False,
            "torch_dtype": torch.float16,
            "num_beams": 3,
            "max_new_tokens": 384,
            "batch_size": 1,
            "temperature": 0.5,
            "top_p": 0.92,
            "top_k": 50,
            "repetition_penalty": 1.15,
            "use_flash_attn": False,
            "enable_cpu_offload": False,
            "max_memory": None
        }

    elif vram_gb < 16:
        return "ultra_vram_half", {
            "name": "ultra_vram_half",
            "description": "VRAM 12-16GB - Precisión media sin cuantización",
            "vram_threshold": 16,
            "use_cuda": True,
            "load_in_4bit": False,
            "load_in_8bit": False,
            "torch_dtype": torch.float16,
            "num_beams": 5,
            "max_new_tokens": 512,
            "batch_size": 1,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 60,
            "repetition_penalty": 1.2,
            "use_flash_attn": False,
            "enable_cpu_offload": False,
            "max_memory": None
        }

    else:
        return "max_performance", {
            "name": "max_performance",
            "description": "VRAM > 16GB - Máximo rendimiento y calidad",
            "vram_threshold": 16,
            "use_cuda": True,
            "load_in_4bit": False,
            "load_in_8bit": False,
            "torch_dtype": torch.bfloat16 if hasattr(torch, 'bfloat16') else torch.float16,
            "num_beams": 7,
            "max_new_tokens": 1024,
            "batch_size": 1,
            "temperature": 0.8,
            "top_p": 0.97,
            "top_k": 100,
            "repetition_penalty": 1.3,
            "use_flash_attn": False,
            "enable_cpu_offload": False,
            "max_memory": None
        }

def manageArgs():
    """
    @Desc: Parse command-line arguments to configure the process.
    @Result: Returns parsed arguments as a Namespace object.
    """
    parser = argparse.ArgumentParser(description="Main process for Image Description.")
    parser.add_argument('--proc', type=str, help="Process type: MODEL, APPLY", default="APPLY")
    parser.add_argument('--yacimiento', type=str, help="Process type: MODEL, APPLY", default="RAMNOUS")
    parser.add_argument('--region', type=str, help="Process type: MODEL, APPLY", default="ÁTICA")
    args = parser.parse_args()

    # Establecer valores por defecto internos
    args.featuresmodel = "VIT"
    args.model = "LLaVA"

    return args


def manageEnv():
    """
    @Desc: Defines environment paths and variables.
    @Result: Returns a dictionary containing environment paths.
    """

    config = configLoader()
    environment = config.get_environment()

    env_data = {}
    for key, value in environment.items():
        if "realPath" in key:
            env_data[key] = value
        else:
            env_data[key] = os.path.join(environment["realPath"], value)

    os.makedirs(env_data['.pycache'], exist_ok=True)
    os.environ['PYTHONPYCACHEPREFIX'] = env_data['.pycache']
    sys.pycache_prefix = env_data['.pycache']
    env_data['systemName'] = socket.getfqdn()
    env_data['modelsCFG'] = config.get_models()
    return env_data

def manageDefaults():
    config = configLoader()
    environment = config.get_defaults()
    return environment

def getConfigs():
    """
    @Desc: Load environment settings, arguments, and hyperparameters.
    @Result: Stores configurations in processControl variables.
    """

    processControl.env = manageEnv()
    processControl.args = manageArgs()
    processControl.defaults = manageDefaults()




