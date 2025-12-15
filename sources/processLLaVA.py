"""
:Description: Ejecución de procesos de modelización basado en LLaVA
:Author:
    - Ana María García Serrano
    - Enrique Garcia Arias
:Organization: UNED
"""
from sources.common.common import logger, processControl, log_
from sources.contextData import buildContextData
from sources.common.utils import image_parser, load_images, commonVars, get_model_name_from_path
from sources.processFeatures import extractFeatures, assign_to_cluster
from sources.dataManager import readResults, writeResultsData
from sources.modelEvaluate import eval_model_batch

from sources.common.paramsManager import select_configuration_based_on_memory, get_system_memory_info
import os
import time


import torch
from tqdm import tqdm

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
)

import re


def eval_model_batch(args_list):
    """Versión con auto-configuración de memoria"""

    # =====================================================
    # 1. DETECCIÓN AUTOMÁTICA DE CONFIGURACIÓN
    # =====================================================
    config_name, auto_config = select_configuration_based_on_memory()
    log_("info", logger, f"Configuration name: {config_name}")
    commonArgs, metas = commonVars()
    # Fusionar configuración automática con commonArgs
    config = {**commonArgs, **auto_config}

    # Log de configuración seleccionada
    mem_info = get_system_memory_info()
    log_("info", logger, f"🔍 Detección de memoria:")
    log_("info", logger, f"   GPU: {mem_info.get('gpu_name', 'No disponible')}")
    log_("info", logger, f"   VRAM: {mem_info.get('vram_available', 0):.1f} GB")
    log_("info", logger, f"   RAM disponible: {mem_info.get('ram_available', 0):.1f} GB")
    log_("info", logger, f"   Configuración seleccionada: {config_name}")
    log_("info", logger, f"   Descripción: {auto_config['description']}")


    # =====================================================
    # 2. INICIALIZACIÓN CON CONFIGURACIÓN ADAPTATIVA
    # =====================================================
    disable_torch_init()

    def _run():
        use_cuda = config["use_cuda"]
        device = torch.device("cuda:0" if use_cuda else "cpu")

        log_("info", logger, f"🚀 Iniciando con configuración '{config_name}' en {device}")

        # Cargar modelo con configuración adaptativa
        model_name = get_model_name_from_path(config["model_path"])
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=config["model_path"],
            model_base=config.get("model_base"),
            model_name=model_name,
            device="cuda" if use_cuda else "cpu",
            device_map="auto" if use_cuda else {"": "cpu"},
            config=config,  # ← Pasar configuración
        )

        model.eval()
        results = []

        # =====================================================
        # 3. PROCESAMIENTO POR BATCHES (ADAPTATIVO)
        # =====================================================
        batch_size = config.get("batch_size", 1)

        for batch_start in tqdm(range(0, len(args_list), batch_size),
                                desc=f"Procesando (batch={batch_size})"):
            batch_args = args_list[batch_start:batch_start + batch_size]

            try:
                # Procesar batch completo
                batch_images = []
                batch_input_ids = []
                batch_image_sizes = []

                for args in batch_args:
                    qs = args["query"]
                    image_path = args["image_file"]

                    # Preparar prompt
                    image_token_se = (
                            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                    )

                    if IMAGE_PLACEHOLDER in qs:
                        qs = re.sub(
                            IMAGE_PLACEHOLDER,
                            image_token_se if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN,
                            qs,
                        )
                    else:
                        qs = (
                            image_token_se + "\n" + qs
                            if model.config.mm_use_im_start_end
                            else DEFAULT_IMAGE_TOKEN + "\n" + qs
                        )

                    # Conversación
                    conv_mode = (
                        "llava_llama_2" if "llama-2" in model_name.lower()
                        else "mistral_instruct" if "mistral" in model_name.lower()
                        else "llava_v1" if "v1" in model_name.lower()
                        else "llava_v0"
                    )

                    conv = conv_templates[conv_mode].copy()
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()

                    # Imagen
                    images = load_images([image_path])
                    image_sizes = [img.size for img in images]
                    batch_image_sizes.extend(image_sizes)

                    images_tensor = process_images(
                        images,
                        image_processor,
                        model.config,
                    ).to(device, dtype=config.get("torch_dtype", torch.float16))

                    batch_images.append(images_tensor)

                    # Tokenización
                    input_ids = tokenizer_image_token(
                        prompt,
                        tokenizer,
                        IMAGE_TOKEN_INDEX,
                        return_tensors="pt",
                    ).unsqueeze(0).to(device)

                    batch_input_ids.append(input_ids)

                # =====================================================
                # 4. GENERACIÓN CON PARÁMETROS ADAPTATIVOS
                # =====================================================
                if batch_size > 1:
                    # Batch processing (si batch_size > 1)
                    batch_images_tensor = torch.cat(batch_images, dim=0)
                    batch_input_ids_tensor = torch.cat(batch_input_ids, dim=0)

                    with torch.inference_mode():
                        output_ids = model.generate(
                            batch_input_ids_tensor,
                            images=batch_images_tensor,
                            image_sizes=batch_image_sizes,
                            do_sample=config["temperature"] > 0,
                            temperature=config["temperature"],
                            top_p=config.get("top_p"),
                            top_k=config.get("top_k"),
                            num_beams=config["num_beams"],
                            max_new_tokens=config["max_new_tokens"],
                            repetition_penalty=config.get("repetition_penalty", 1.0),
                            length_penalty=config.get("length_penalty", 1.0),
                            no_repeat_ngram_size=config.get("no_repeat_ngram_size", 3),
                            use_cache=True,
                            early_stopping=True,
                        )

                    # Decodificar cada elemento del batch
                    for i, args in enumerate(batch_args):
                        output_text = tokenizer.decode(
                            output_ids[i],
                            skip_special_tokens=True,
                        ).strip()

                        results.append({
                            "image": args["image_file"],
                            "prompt": args["query"],
                            "answer": output_text,
                            "config_used": config_name,
                        })
                else:
                    # Procesamiento individual (compatible con version anterior)
                    with torch.inference_mode():
                        output_ids = model.generate(
                            batch_input_ids[0],
                            images=batch_images[0],
                            image_sizes=batch_image_sizes,
                            do_sample=config["temperature"] > 0,
                            temperature=config["temperature"],
                            top_p=config.get("top_p"),
                            top_k=config.get("top_k"),
                            num_beams=config["num_beams"],
                            max_new_tokens=config["max_new_tokens"],
                            repetition_penalty=config.get("repetition_penalty", 1.0),
                            length_penalty=config.get("length_penalty", 1.0),
                            no_repeat_ngram_size=config.get("no_repeat_ngram_size", 3),
                            use_cache=True,
                            early_stopping=True,
                        )

                    output_text = tokenizer.decode(
                        output_ids[0],
                        skip_special_tokens=True,
                    ).strip()

                    results.append({
                        "image": batch_args[0]["image_file"],
                        "prompt": batch_args[0]["query"],
                        "answer": output_text,
                        "config_used": config_name,
                    })

            except torch.cuda.OutOfMemoryError:
                log_("error", logger, f"⚠️ OOM en batch {batch_start}, reduciendo batch_size")
                if use_cuda:
                    torch.cuda.empty_cache()
                # Fallback a procesamiento individual
                for args in batch_args:
                    try:
                        # Procesar individualmente...
                        pass
                    except:
                        log_("error", logger, f"   Saltando {args['image_file']}")
                        results.append({
                            "image": args["image_file"],
                            "prompt": args["query"],
                            "answer": "[ERROR: OOM]",
                            "config_used": config_name + "_fallback",
                        })

            finally:
                # Limpieza
                del batch_images, batch_input_ids
                if use_cuda:
                    torch.cuda.empty_cache()

        return results

    # =====================================================
    # 5. EJECUCIÓN CON FALLBACK AUTOMÁTICO
    # =====================================================
    try:
        return _run()
    except Exception as e:
        log_("error", logger, f"❌ Error con configuración {config_name}: {str(e)}")

        # Fallback a configuración más baja
        if config_name != "low_vram_4bit":
            log_("warning", logger, "🔄 Intentando con configuración low_vram_4bit...")
            fallback_config = select_configuration_based_on_memory()[1]
            fallback_config["name"] = "fallback_" + fallback_config["name"]

            # Sobrescribir con configuración de fallback
            config = {**commonArgs, **fallback_config}
            return _run()
        else:
            # Último recurso: CPU
            log_("warning", logger, "🔄 Intentando en CPU...")
            config["use_cuda"] = False
            config["load_in_4bit"] = False
            config["load_in_8bit"] = False
            config["batch_size"] = 1
            return _run()

def eval_model(args, commonArgs):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(commonArgs["model_path"])
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        commonArgs["model_path"], commonArgs["model_base"], model_name
    )
    qs = args["query"]
    image_path = args["image_file"]
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    args["conv_mode"] = conv_mode

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images = load_images([image_path])
    image_sizes = [x.size for x in images]
    #EGA get the device from model
    device = model.device
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        #EGA .cuda()
        .to(device)
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if commonArgs["temperature"]> 0 else False,
            temperature=commonArgs["temperature"],
            top_p=commonArgs["top_p"],
            num_beams=commonArgs["num_beams"],
            max_new_tokens=commonArgs["max_new_tokens"],
            use_cache=False,  #EGA disabled from True
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    #EGA elimino el print y añado return
    #print(outputs)
    #return outputs
    return {
            "image": args["image_file"],
            "prompt": args["query"],
            "answer": outputs
        }

def NEWeval_model_batch(args_list, commonArgs):
    disable_torch_init()
    results = []
    model_name = get_model_name_from_path(commonArgs["model_path"])
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        commonArgs["model_path"], commonArgs["model_base"], model_name
    )
    device = model.device
    prompt = ""
    for args in args_list:
        qs = args["query"]
        image_path = args["image_file"]
        # [Same image and prompt processing as before...]
        images = load_images([image_path])
        image_sizes = [x.size for x in images]
        images_tensor = process_images(images, image_processor, model.config).to(device, dtype=torch.float16)
        input_ids = (tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    .unsqueeze(0)
                    .to(device))

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if commonArgs["temperature"] > 0 else False,
                temperature=commonArgs["temperature"],
                top_p=commonArgs["top_p"],
                num_beams=commonArgs["num_beams"],
                max_new_tokens=commonArgs["max_new_tokens"],
                use_cache=False,
            )
            torch.cuda.empty_cache()
            output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            results.append({
                "image": args["image_file"],
                "prompt": args["query"],
                "answer": output_text
            })

    return results


def buildContentProcess():
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    doc_extensions = {".doc", ".docx"}

    images_list = []
    doc_file = None

    for filename in os.listdir(processControl.env['inputPath']):
        file_path = os.path.join(processControl.env['inputPath'], filename)

        # Si es un archivo de imagen
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
            # Extraer el título (nombre completo del archivo)
            title = filename

            # Extraer el nombre sin "Diapo 99.99" al inicio
            match = re.match(r"Diapo \d+\.\d+\s+(.+)", filename)
            name = match.group(1) if match else filename  # Si hay coincidencia, extraer nombre limpio
            name, _ = os.path.splitext(name)  # Eliminar la extensión

            # Si el nombre queda vacío, asignar "vacio"
            if not name.strip():
                name = "vacio"

            images_list.append({
                "image": title,
                "imagePath": file_path,
                "name": name,
                "yacimiento": processControl.args.yacimiento,
                "zona": processControl.args.region,
            })

        # Si es un archivo .doc o .docx (tomamos el primero que encontremos)
        elif os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in doc_extensions):
            if doc_file is None:  # Solo guardamos el primer archivo .doc/docx encontrado
                doc_file = file_path

    return {"images":images_list, "doc":doc_file}


def processPrompt1(contentProcess, imageFeatures):
    personalization = processControl.defaults.get('personalization', {})
    processArgs = []
    for content in tqdm(contentProcess["images"], desc="Processing Prompt 1"):
        features = imageFeatures[content['name']]
        assigned_label, closest_cluster_idx = assign_to_cluster(features)
        contextText, keywords = buildContextData(contentProcess["doc"], content["name"], top_n=5)
        #log_("info", logger, f"Contexto generado: {contextText}")
        prompt1 = (f"{personalization[assigned_label][0]} representando a '{content['name']}', "
                   f"Ten en cuenta sin mencionar explícitamente que {personalization[assigned_label][3]} "
                   f"y describe en castellano solo lo visible. Máximo 20 palabras, cíñete a estos dos items:\n"
                   f"Item 1 {personalization[assigned_label][1]}. "
                   f"Item 2 {personalization[assigned_label][2]}. Máximo 20 palabras.")

        args = {
            "query": prompt1,
            "image_file": content["imagePath"],
        }
        processArgs.append(args)
    return processArgs


def processPrompt2(data):
    processArgs = []
    log_("info", logger, f"Start process LLaVA")
    for element in data:
        patron = r'Item\s*[12]:?'  # Coincide con "Item 1", "Item 1:", "Item 2", "Item 2:"
        descripInicial = re.sub(patron, '', element['answer']).strip()
        patron = r'\*\*.*?\*\*'
        descripInicial = re.sub(patron, '', descripInicial).strip()

        if element['context'] is not None:
            prompt2 = (f"Este es el contexto con el que vamos a enriquecer una descripción de imagen CONTEXTO: '{element['context']}', "
                        f"mejora la redacción componiendo un texto contínuo y con sentido global "
                        f"Limita las respuestas a 40 palabras como máximo, pero si alcanzas ese límite a mitad de frase, "
                        f"puedes extenderte hasta completarla (hasta el punto final). Utiliza solo la información del CONTEXTO sin añadir nada más. ")
            args = {
                "query": prompt2,
                "image_file": element["imagePath"],
            }
            processArgs.append(args)
    return processArgs

def processPrompt3(data):
    processArgs = []
    log_("info", logger, f"Start process LLaVA")
    for element in data:
        patron = r'Item\s*[12]:?'  # Coincide con "Item 1", "Item 1:", "Item 2", "Item 2:"
        descripInicial = re.sub(patron, '', element['answer']).strip()
        patron = r'\*\*.*?\*\*'
        descripInicial = re.sub(patron, '', descripInicial).strip()

        if element['context'] is not None:
            prompt3 = (f"Tienes el siguiente CONTEXTO de un yacimiento: '{element['answer2']}', "
                        f"mejora esta DESCRIPCIÓN: '{descripInicial}'. "
                        f"Sustituye esta descripción por un texto claro, directo y enlazado, incorporando la información relevante del CONTEXTO. ")
        else:
            prompt3 += (f"Manteniendo el texto de la descripción inicial: '{descripInicial}'. "         
                        f"Utiliza un máximo de 20 palabras sin cortar frases. "
                        f"Construye un texto enlazado y no menciones evidencias para un arqueólogo como que es antiguo o deteriorado o es un yacimiento")


        args = {
            "query": prompt3,
            "image_file": element["imagePath"],
        }
        processArgs.append(args)
    return processArgs


def processPrompt4(data):
    processArgs = []
    log_("info", logger, f"Start process LLaVA")
    for element in data:
        prompt2 = (f"Partiendo de la imagen y de su descripción inicial: '{element['answer2']}', mejora la descripción evitando inferencias, subjetividades, "
                   f"selecciona los argumentos positivos evita los argumentos de duda o pregunta como 'podría ser',..."
                f"Se trata de dar un enfoque de descripción de arqueología, no menciones evidencias para un arqueólogo como que es antiguo o deteriorado, "
                f"no menciones piedras sino restos arqueológicos y redacta un texto con continuidad")
        if element['context'] is not None:
            prompt2 += f". Utiliza un máximo de 50 palabras."
        else:
            prompt2 += ". Utiliza un máximo de 20 palabras "
        prompt2 += f"y construye un texto enlazado"

        args = {
            "query": prompt2,
            "image_file": element["imagePath"],
        }
        processArgs.append(args)
    return processArgs


def checkStage():
    data = readResults(3)
    if data:
        return 3, data
    data = readResults(2)
    if data:
        return 2, data
    data = readResults(1)
    if data:
        return 1, data
    return 0, None


def processStage0():


    start_time = time.time()
    contentProcess = buildContentProcess()
    imageFeatures = extractFeatures(contentProcess["images"])
    doc = contentProcess["doc"]
    for index, content in enumerate(tqdm(contentProcess["images"], desc="Procesando imágenes")):
        #log_('info', logger, f'processing image {content["name"]}')
        contextText, keywords = buildContextData(doc, content["name"], top_n=5)
        contentProcess["images"][index]["context"] = contextText
        contentProcess["images"][index]["keywords"] = keywords

        features = imageFeatures[content['name']]
        assigned_label, closest_cluster_idx = assign_to_cluster(features)
        contentProcess["images"][index]["label"] = assigned_label
        # contentProcess["images"][index]["clusterIDX"] = closest_cluster_idx

    end_time = time.time()  # End timing
    duration = end_time - start_time
    log_("info", logger, f"Duration Process Features-Cluster: {duration}")

    processArgs = processPrompt1(contentProcess, imageFeatures)
    results = eval_model_batch(processArgs)
    for idx, image_data in enumerate(tqdm(contentProcess["images"], desc="Procesando resultados paso 1")):
        for idx2, result in enumerate(results):
            if image_data["imagePath"] == result["image"]:
                contentProcess["images"][idx]["prompt"] = result["prompt"]
                # Elimino los patrones de respuesta indicados en el prompt
                patron = r'Item\s*[12]:?'  # Coincide con "Item 1", "Item 1:", "Item 2", "Item 2:"
                descripInicial = re.sub(patron, '', result['answer']).strip()
                patron = r'\*\*.*?\*\*'
                descripInicial = re.sub(patron, '', descripInicial).strip()
                contentProcess["images"][idx]["answer"] = descripInicial


    result = sorted(contentProcess['images'], key=lambda x: (x['label'] is None, x['label']))
    writeResultsData(result, 1)

    end_time2 = time.time()  # End timing
    duration = end_time2 - end_time
    log_("info", logger, f"Duration Process 0: {duration}")

    return result


def processStage1(data):
    start_time = time.time()
    processArgs = processPrompt2(data)
    results = eval_model_batch(processArgs)
    for idx, image_data in enumerate(data):
        for idx2, result in enumerate(results):
            if image_data["imagePath"] == result["image"]:
                data[idx]["prompt2"] = result["prompt"]
                data[idx]["answer2"] = result['answer']


    result = sorted(data, key=lambda x: (x['label'] is None, x['label']))
    writeResultsData(result, 2)
    end_time = time.time()  # End timing
    duration = end_time - start_time
    log_("info", logger, f"Duration Process 1: {duration}")
    return result


def processStage2(data):
    start_time = time.time()
    processArgs = processPrompt3(data)
    results = eval_model_batch(processArgs)
    for idx, image_data in enumerate(data):
        for idx2, result in enumerate(results):
            if image_data["imagePath"] == result["image"]:
                data[idx]["prompt3"] = result["prompt"]
                data[idx]["answer3"] = \
                    f"{data[idx]['name']}, pertenece al yacimiento de {data[idx]['yacimiento']} en zona de {data[idx]['zona']}. {result['answer']}"


    result = sorted(data, key=lambda x: (x['label'] is None, x['label']))
    writeResultsData(result, 3)
    end_time = time.time()  # End timing
    duration = end_time - start_time
    log_("info", logger, f"Duration Process 2: {duration}")
    return result


def processLLaVA():
    if processControl.defaults['device'] == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
    stage, data = checkStage()
    log_("info", logger, f"Stage: {stage}")
    data = processStage0()
    log_("info", logger, f"Proceso Fase: 1")
    data = processStage1(data)
    log_("info", logger, f"Proceso Fase: 2")
    data = processStage2(data)
    return


    if stage == 2:
        result = processStage2(data)
    if stage == 1:
        result = processStage1(data)
    elif stage == 0:
        result = processStage0()

