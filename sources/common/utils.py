from sources.common.common import logger, processControl, log_
import json

import time
import os
from os.path import isdir
from huggingface_hub import login
import requests
from PIL import Image
from io import BytesIO


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def mkdir(dir_path):
    """
    @Desc: Creates directory if it doesn't exist.
    @Usage: Ensures a directory exists before proceeding with file operations.
    """
    if not isdir(dir_path):
        os.makedirs(dir_path)


def dbTimestamp():
    """
    @Desc: Generates a timestamp formatted as "YYYYMMDDHHMMSS".
    @Result: Formatted timestamp string.
    """
    timestamp = int(time.time())
    formatted_timestamp = str(time.strftime("%Y%m%d%H%M%S", time.gmtime(timestamp)))
    return formatted_timestamp

class configLoader:
    """
    @Desc: Loads and provides access to JSON configuration data.
    @Usage: Instantiates with path to config JSON file.
    """
    def __init__(self, config_path='config.json'):
        self.base_path = os.path.realpath(os.getcwd())
        realConfigPath = os.path.join(self.base_path, config_path)
        self.config = self.load_config(realConfigPath)

    def load_config(self, realConfigPath):
        with open(realConfigPath, 'r') as config_file:
            return json.load(config_file)

    def get_environment(self):
        environment =  self.config.get("environment", None)
        environment["realPath"] = self.base_path
        return environment

    def get_defaults(self):
        return self.config.get("defaults", {})

    def get_models(self):
        return self.config.get("models", {})

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def OLDload_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def load_images(image_paths):
    images = [Image.open(path).resize((224, 224)) for path in image_paths]  # Resize to 224x224
    return images


def buildImageProcess(DirectoryPath=None):
    result = []
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    for image_name in os.listdir(DirectoryPath):
        if os.path.splitext(image_name)[1].lower() in supported_extensions:
            result.append({"imagePath": os.path.join(DirectoryPath, image_name), "name": image_name})
    return result


def huggingface_login():
    try:
        # Add your Hugging Face token here, or retrieve it from environment variables
        token = processControl.defaults['token'] if 'token' in processControl.defaults else ['', '']
        login(token)
        print("Successfully logged in to Hugging Face.")
    except Exception as e:
        print("Error logging into Hugging Face:", str(e))
        raise


def commonVars():
    model_path = "liuhaotian/llava-v1.5-7b"
    commonArgs = {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "conv_mode": None,
        "sep": ",",
        "temperature": 0.2,  # 0
        "top_p": None,
        "num_beams": 3,  # 1
        "max_new_tokens": 128  # 512, 256
    }
    metas = {
        "yacimiento": processControl.args.yacimiento,
        "region": processControl.args.region,
    }
    """
    metas = {
        "yacimiento": "RAMNOUS",
        "region": "ÁTICA"
    }    
    """

    personalization = {
        "Panorámica": ["Para esta fotografía panorámica",
                       "**Ubicación y entorno**: Describe el paisaje y el tipo de terreno",
                       "**Elemento principal yacimiento arqueológico**: Explica la estructura, y su disposición",
                       "es un yacimiento arqueológico en su vista general y amplia no son sólo piedras"],
        "Dibujos": ["Para este dibujo que muestra con detalle un elemento o una estructura",
                    "**Composición y contorno**: Describe su estructura, composición, apariencia",
                    "**Elemento principal**: Explica qué simboliza o representa culturalmente",
                    "es la representación de una estructura arqueológica singular y que puede estar incompleta"],
        "Detalles": ["Para esta fotografía que enfoca un detalle",
                     "**Ubicación y entorno**: Describe el entorno y cómo se ubica el elemento principal",
                     "**Elemento principal que protagoniza la imagen**: Explica la estuctura de el elemento principal",
                     "es un yacimiento arqueológico con su elemento principal no son sólo piedras"],
        "Diapositivas": ["Para esta fotografía de exposición",
                         "**Composición**: Describe su composición y contorno",
                         "**Elemento principal**: Explica sus características",
                         "es un objeto arqueológico de valor singular"]
    }
    return commonArgs, metas, personalization

