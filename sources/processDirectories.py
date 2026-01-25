"""
:Description: Módulo para el preproceso y postproceso de directorios y ficheros
:Author:
    - Ana María García Serrano
    - Enrique Garcia Arias
:Organization: UNED
"""


from sources.common.common import logger, processControl, log_

import os
import shutil
import json


def procesar_yacimiento(nomYacimiento):
    nomYacimiento = processControl.args.yacimiento
    print(f"--- Iniciando proceso para {nomYacimiento} ---")
    filePath = {
        'output': processControl.env.get("outputPath", ""),
        'input': processControl.env.get("inputPath", ""),
        'store': processControl.env.get("storePath", "")
    }

    # 1) se sitúa en filePath.output
    path_output = filePath['output']

    # 2) abre el fichero result_3.json
    json_path = os.path.join(path_output, 'result_3.json')

    if not os.path.exists(json_path):
        print(f"Error: No se encuentra el archivo {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 3) del primer registro almacena el valor de registro['zona'] en la variable yacimientoOLD
    if not data:
        print("Error: El JSON está vacío.")
        return
    yacimientoOLD = data[0]['zona']
    print(f"Zona detectada (yacimientoOLD): {yacimientoOLD}")

    # 4) se sitúa en filePath.output.PELOPONESO
    # 5) crea el directorio con el nombre yacimientoOLD
    path_peloponeso = os.path.join(path_output, 'PELOPONESO')
    target_dir = os.path.join(path_peloponeso, yacimientoOLD)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Directorio creado: {target_dir}")

    # 6) mueve todos los archivos de filePath.input al directorio creado
    path_input = filePath['input']
    if os.path.exists(path_input):
        for archivo in os.listdir(path_input):
            src_file = os.path.join(path_input, archivo)
            if os.path.isfile(src_file):
                shutil.move(src_file, target_dir)

    # 7) copia el archivo filePath.output.result_3.json al directorio creado
    shutil.copy(json_path, os.path.join(target_dir, 'result_3.json'))

    # 8) borra todos los archivos *.json del directorio filePath.output
    for archivo in os.listdir(path_output):
        if archivo.endswith('.json'):
            os.remove(os.path.join(path_output, archivo))

    # 9) se sitúa en filePath.input
    # 10) borra todos los archivos en ese directorio (limpieza de restos)
    if os.path.exists(path_input):
        for archivo in os.listdir(path_input):
            archivo_full = os.path.join(path_input, archivo)
            if os.path.isfile(archivo_full):
                os.remove(archivo_full)

    # 11) se sitúa en filePath.store.YACIMIENTOS.INFORMES
    path_informes = os.path.join(filePath['store'], 'YACIMIENTOS', 'INFORMES')

    # 12) localiza un fichero con nombre <nomYacimiento>.docx (Ya convertido manualmente)
    # 13) [ELIMINADO] Conversión
    # 14) copia ese fichero .docx al directorio filePath.input

    docx_name = f"{nomYacimiento}.docx"
    docx_path = os.path.join(path_informes, docx_name)

    if os.path.exists(docx_path):
        shutil.copy(docx_path, path_input)
        print(f"Copiado informe: {docx_name} al directorio input.")
    else:
        print(f"Advertencia: No se encontró el informe {docx_path}")

    # 15) se sitúa en filePath.store.YACIMIENTOS.IMAGENES.<nomYacimiento>
    path_imagenes = os.path.join(filePath['store'], 'YACIMIENTOS', 'IMAGENES', nomYacimiento)

    if os.path.exists(path_imagenes):
        # 16) copia todos los archivos.jpg al directorio filePath.input
        count_jpg = 0
        for archivo in os.listdir(path_imagenes):
            if archivo.lower().endswith('.jpg'):
                src_file = os.path.join(path_imagenes, archivo)
                shutil.copy(src_file, path_input)
                count_jpg += 1
        print(f"Se copiaron {count_jpg} imágenes .jpg al input.")

    print("--- Proceso finalizado ---")
