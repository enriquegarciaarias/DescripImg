"""
@Purpose: Main script for initializing environment settings and start procesing the Image Captioning project, handling main modes:
@Usage: Run `python mainProcess.py`.
"""

from sources.common.common import processControl, logger, log_
from sources.common.utils import huggingface_login
from sources.common.paramsManager import getConfigs

import torch
import os


def mainProcess():
    """
    Main process for handling either model training or applying a trained model to new data.

    This function serves as the entry point for executing the main tasks in the pipeline. Depending on the operation mode specified
    (`MODEL` or `APPLY`), it either extracts features, trains a model, and saves it, or applies an existing trained model to new data.

    The function operates in two modes:
    - **MODEL mode**: Extracts features, trains the model, and saves it.
    - **APPLY mode**: Applies the trained model to new data for inference.

    :return: A boolean indicating the success of the main process.
    :rtype: bool
    """
    # Set model and pretrained dataset
    processControl.process['modelName'] = "ViT-L/14"
    processControl.process['pretrainedDataset'] = "laion2b_s32b_b82k"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_("info", logger, f"procesing: {processControl.args.proc}, MODEL: {processControl.args.model}")
    processControl.defaults['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    log_("info", logger, f"Using device: {processControl.defaults['device']}")

    # MODEL mode: Extract features, train the model, and save it
    if processControl.args.proc == "MODEL":
        from sources.processFeatures import processFeatures
        featuresFile, imagesLabels = processFeatures()

    # APPLY mode: Apply the trained model to new data
    if processControl.args.proc == "APPLY":
        huggingface_login()
        if processControl.args.model == "LLM":
            from sources.processLLM import processLlama2
            processLlama2()
        elif processControl.args.model == "MISTRAL":
            from sources.processLLM import processMistral
            processMistral()
        elif processControl.args.model == "LLaVA":
            from sources.processLLaVA import processLLaVA
            processLLaVA()


    if processControl.args.proc == "EVAL":
        from sources.evaluacion import procesoEvaluacion
        procesoEvaluacion()

    return True


if __name__ == '__main__':
    """
    Entry point for starting the main image caption process.

    This block of code is executed when the script is run directly. It logs the start of the process, retrieves configuration settings,
    and then triggers the main process. After the main process completes, it logs the completion of the task.

    The function performs the following steps:
    - Logs the start of the process.
    - Calls `getConfigs()` to retrieve necessary configurations.
    - Executes `mainProcess()` to handle model training or application.
    - Logs the completion of the process.

    :return: None
    """

    log_("info", logger, "********** STARTING Main Image Caption Process **********")
    getConfigs()
    log_("info", logger, f"System Name: {processControl.env['systemName']}")
    if processControl.env['systemName'] == "tesla.informatica.uned.es":
        import torch.distributed as dist
        dist.init_process_group(backend='nccl')
        log_("info", logger, "This is TESLA")
        mainProcess()
        dist.destroy_process_group()
    elif processControl.env['systemName'] == "PULSAR-PRO":
        log_("info", logger, "This is PULSAR-PRO")
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.cuda.set_per_process_memory_fraction(0.98, device=0)
        torch.backends.cuda.max_split_size_mb = 64
        mainProcess()

    else:
        log_("info", logger, "This is Local")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        mainProcess()

    log_("info", logger, "********** PROCESS COMPLETED **********")
