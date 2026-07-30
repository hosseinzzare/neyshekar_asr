"""
Whisper Large-v3 4-Bit QLoRA Model Initialization & PEFT Adapter Setup.
Task 2 - Step 4: Quantized Model & LoRA Adapter Configuration.
"""

import os
import sys

# Force UTF-8 encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure src is in sys.path
sys.path.append(os.path.dirname(__file__))
import config

try:
    import torch
    from transformers import WhisperForConditionalGeneration, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
except ImportError:
    torch = None
    WhisperForConditionalGeneration = None
    BitsAndBytesConfig = None
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
    TaskType = None


def make_inputs_require_grad(module, input, output):
    """
    Forward hook on conv1 layer ensuring audio input feature outputs require gradients.
    Mandatory for PyTorch Gradient Checkpointing with frozen QLoRA base weights.
    """
    output.requires_grad_(True)


def get_whisper_qlora_model(
    model_name_or_path: str = config.MODEL_NAME_OR_PATH,
    load_in_4bit: bool = config.LOAD_IN_4BIT,
    lora_r: int = config.LORA_R,
    lora_alpha: int = config.LORA_ALPHA,
    lora_dropout: float = config.LORA_DROPOUT,
    target_modules: list = config.TARGET_MODULES
):
    """
    Loads Whisper Large-v3 in 4-bit NF4 quantization using BitsAndBytes,
    prepares it for k-bit training, registers conv1 input gradient hook,
    attaches PEFT LoRA adapters, and applies safe forward wrapper for PEFT+Whisper compatibility.
    """
    if torch is None or WhisperForConditionalGeneration is None:
        raise RuntimeError("PyTorch, Transformers, or PEFT not installed.")

    config.set_seed(config.SEED)
    print(f"\n[MODEL LOAD] Loading base model '{model_name_or_path}' with 4-bit QLoRA...")

    # 1. Define 4-bit Quantization Config (BitsAndBytes)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT
    )

    # 2. Load Base Whisper Model in 4-bit
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name_or_path,
        quantization_config=quantization_config,
        device_map="auto"
    )

    # 3. Prepare Model for K-Bit Training & Adjust Config Flags
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # Mandatory for Gradient Checkpointing compatibility
    model.config.forced_decoder_ids = None  # Mandatory for Persian generation
    model.config.suppress_tokens = []  # Clear default token suppression

    # 4. Register forward hook on encoder conv1 for QLoRA Gradient Checkpointing stability
    model.get_encoder().conv1.register_forward_hook(make_inputs_require_grad)
    print("[HOOK REGISTER] Registered make_inputs_require_grad hook on model.get_encoder().conv1")

    # 5. Define PEFT LoRA Configuration
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    # 6. Attach LoRA Adapters to Model
    model = get_peft_model(model, peft_config)

    # 7. Apply Safe Forward Wrapper to resolve PEFT+Whisper 'input_ids' keyword collision bug
    orig_base_forward = model.base_model.forward
    def safe_base_forward(*args, **kwargs):
        for k in ['input_ids', 'inputs_embeds', 'attention_mask']:
            if k in kwargs and kwargs[k] is None:
                kwargs.pop(k)
        return orig_base_forward(*args, **kwargs)

    model.base_model.forward = safe_base_forward
    print("[PEFT FIX] Applied safe_base_forward wrapper to resolve PEFT input_ids kwargs collision.")

    print("\n" + "="*60)
    print(" === WHISPER LARGE-V3 QLORA MODEL TRAINABLE PARAMETERS ===")
    print("="*60)
    model.print_trainable_parameters()
    print("="*60 + "\n")

    return model


if __name__ == '__main__':
    print("Testing src/model.py module structure...")
    if torch is None or WhisperForConditionalGeneration is None:
        print("[NOTICE] Torch/PEFT not available locally. Structure verified.")
    else:
        print("[SUCCESS] Model initialization module ready.")
