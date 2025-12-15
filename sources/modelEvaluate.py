"""
:Description: Módulo para realizar las evaluaciones con el modelo de lenguaje
:Author:
    - Ana María García Serrano
    - Enrique Garcia Arias
:Organization: UNED
"""
from sources.common.common import logger, processControl, log_
from sources.common.utils import load_images, get_model_name_from_path

from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
)
import re
import torch
from tqdm import tqdm
#LLaVA constants
CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15
LOGDIR = "."
# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

def eval_model_batch(args_list, commonArgs):
    disable_torch_init()

    # =====================================================
    # INTERNAL RUNNER (single-device execution)
    # =====================================================
    def _run(device_str):
        use_cuda = (device_str == "cuda" and torch.cuda.is_available())
        device = torch.device("cuda:0" if use_cuda else "cpu")

        log_("info", logger, f"Running inference on device: {device}")

        # ============================
        # LOAD MODEL (ONCE)
        # ============================
        model_name = get_model_name_from_path(commonArgs["model_path"])
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=commonArgs["model_path"],
            model_base=commonArgs.get("model_base"),
            model_name=model_name,
            device="cuda" if use_cuda else "cpu",
            device_map="auto" if use_cuda else {"": "cpu"},
        )

        model.eval()
        results = []

        # ============================
        # MAIN LOOP (SEQUENTIAL = SAFE)
        # ============================
        for args in tqdm(args_list, desc=f"Processing samples ({device_str})"):
            input_ids = None
            images_tensor = None
            output_ids = None

            try:
                qs = args["query"]
                image_path = args["image_file"]

                # ---------- Prompt ----------
                image_token_se = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
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

                # ---------- Conversation ----------
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

                # ---------- Image ----------
                images = load_images([image_path])
                image_sizes = [img.size for img in images]

                images_tensor = process_images(
                    images,
                    image_processor,
                    model.config,
                ).to(
                    device,
                    dtype=torch.float16 if use_cuda else torch.float32,
                )

                # ---------- Tokenization ----------
                input_ids = (
                    tokenizer_image_token(
                        prompt,
                        tokenizer,
                        IMAGE_TOKEN_INDEX,
                        return_tensors="pt",
                    )
                    .unsqueeze(0)
                    .to(device)
                )

                # ---------- Generation ----------
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=images_tensor,
                        image_sizes=image_sizes,
                        do_sample=commonArgs["temperature"] > 0,
                        temperature=commonArgs["temperature"],
                        top_p=commonArgs["top_p"],
                        num_beams=commonArgs["num_beams"],
                        max_new_tokens=commonArgs["max_new_tokens"],
                        use_cache=True,
                    )

                output_text = tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=True,
                ).strip()

                results.append(
                    {
                        "image": image_path,
                        "prompt": args["query"],
                        "answer": output_text,
                    }
                )

            except torch.cuda.OutOfMemoryError:
                log_("error", logger, f"⚠️ CUDA OOM on {image_path}, skipping sample")
                if use_cuda:
                    torch.cuda.empty_cache()

            finally:
                # ---------- HARD CLEANUP ----------
                del input_ids, images_tensor, output_ids
                if use_cuda:
                    torch.cuda.empty_cache()

        return results

    # =====================================================
    # ORCHESTRATION WITH FALLBACK
    # =====================================================
    prefer_cuda = (
        processControl.defaults.get("device") == "cuda"
        and torch.cuda.is_available()
    )

    if prefer_cuda:
        try:
            return _run("cuda")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "cuda" in str(e).lower():
                log_("error", logger, "🔥 CUDA failed globally, falling back to CPU")
                torch.cuda.empty_cache()
            else:
                raise

    # ============================
    # CPU FALLBACK (FINAL)
    # ============================
    return _run("cpu")




def OLDeval_model_batch(args_list, commonArgs):
    disable_torch_init()
    results = []

    # Load model and tokenizer once
    model_name = get_model_name_from_path(commonArgs["model_path"])
    log_("info", logger, f"Evaluating {commonArgs['model_path']}")
    if processControl.defaults['device'] == "cpu":
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            commonArgs["model_path"],
            commonArgs["model_base"],
            model_name,
            load_in_8bit=False,  # Disable 8-bit quantization
            load_in_4bit=False,  # Disable 4-bit quantization
            device_map="cpu", # Explicitly map to CPU
            device="cpu"
        )

    elif processControl.defaults['device'] == "cuda":
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            commonArgs["model_path"],
            commonArgs["model_base"],
            model_name,
            load_8bit=False,
            load_4bit=True,
            device_map="auto",
            device="cuda",
        )

    device = model.device
    log_("info", logger, f"Using device: {device}")
    batch_input_ids = []
    batch_images = []
    batch_image_sizes = []
    batch_prompts = []
    batch_metadata = []

    for args in tqdm(args_list, desc="Building args list"):
        qs = args["query"]
        image_path = args["image_file"]

        # Process prompt
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

        # Conversation mode selection
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

        # Process image
        images = load_images([image_path])  # Load as list
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            image_processor,
            model.config
        ).to(device, dtype=torch.float16)

        # Tokenize input
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(device)
        )

        batch_input_ids.append(input_ids)
        batch_images.append(images_tensor)
        batch_image_sizes.append(image_sizes)
        batch_prompts.append(prompt)
        batch_metadata.append(args)

    # Process each image-prompt pair sequentially (if memory is limited)
    for i in tqdm(range(len(batch_input_ids)), desc="Processing Batches", unit="batch"):
        with torch.inference_mode():
            output_ids = model.generate(
                batch_input_ids[i],
                images=batch_images[i],
                image_sizes=batch_image_sizes[i],
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
            "image": batch_metadata[i]["image_file"],
            "prompt": batch_metadata[i]["query"],
            "answer": output_text
        })

    return results