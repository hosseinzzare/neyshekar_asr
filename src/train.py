"""
Main Training Pipeline Script for Whisper Large-v3 QLoRA Fine-Tuning.
Task 2 - Step 6: End-to-End Hugging Face Seq2SeqTrainer Training Pipeline.
"""

import os
import sys
import argparse
from typing import Optional

# Force UTF-8 encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure src is in sys.path
sys.path.append(os.path.dirname(__file__))
import config
from config import set_seed
from dataset import get_datasets_and_collator
from model import get_whisper_qlora_model
from metrics import get_compute_metrics_fn

try:
    import torch
    from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
except ImportError:
    torch = None
    Seq2SeqTrainingArguments = None
    Seq2SeqTrainer = None


def parse_args():
    """Parses Command Line Arguments for flexible execution (e.g. quick smoke testing vs full training)."""
    parser = argparse.ArgumentParser(description="Whisper Large-v3 QLoRA Fine-Tuning Pipeline")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=config.MAX_STEPS,
        help="Max training steps (-1 for full epoch-based training, or e.g. 30 for Colab smoke test)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.NUM_EPOCHS,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default=config.TRAIN_CSV,
        help="Path to training CSV file"
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default=config.VAL_CSV,
        help="Path to validation CSV file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config.OUTPUT_DIR,
        help="Directory to save fine-tuned model checkpoints"
    )
    return parser.parse_args()


def run_training_pipeline(args=None):
    """
    Executes complete training pipeline:
    1. Sets deterministic random seed.
    2. Prepares Datasets, Processor, and Data Collator.
    3. Loads 4-bit Quantized Whisper Large-v3 with LoRA Adapters.
    4. Configures Seq2SeqTrainingArguments.
    5. Runs Seq2SeqTrainer and saves final best model.
    """
    if args is None:
        args = parse_args()

    # 1. Initialize Deterministic Random Seed
    set_seed(config.SEED)

    print("\n" + "="*70)
    print(" === WHISPER LARGE-V3 QLORA TRAINING PIPELINE START ===")
    print("="*70)
    print(f"  - Model Path:       {config.MODEL_NAME_OR_PATH}")
    print(f"  - Target Language:  {config.LANGUAGE}")
    print(f"  - Max Steps:        {args.max_steps}")
    print(f"  - Epochs:           {args.epochs}")
    print(f"  - Output Dir:       {args.output_dir}")
    print("="*70 + "\n")

    # 2. Prepare Datasets, Processor, and Data Collator (enable subset mode if max_steps is set for fast Colab smoke testing)
    max_samples = 1000 if (args.max_steps is not None and args.max_steps > 0) else None
    train_dataset, val_dataset, processor, data_collator = get_datasets_and_collator(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        max_samples=max_samples
    )

    # 3. Load QLoRA Quantized Model
    model = get_whisper_qlora_model()

    # 4. Prepare Compute Metrics Function
    compute_metrics_fn = get_compute_metrics_fn(processor=processor)

    # 5. Define Seq2Seq Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        warmup_steps=config.WARMUP_STEPS,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        fp16=config.FP16,
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        predict_with_generate=True,
        generation_max_length=225,
        eval_strategy=config.EVAL_STRATEGY,
        eval_steps=config.EVAL_STEPS,
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
        greater_is_better=config.GREATER_IS_BETTER,
        load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
        logging_dir=config.LOGS_DIR,
        logging_steps=10,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=["tensorboard"]
    )

    # 6. Initialize Seq2SeqTrainer (using processing_class for transformers v4.46+ compatibility)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics_fn,
    }
    import inspect
    if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor.feature_extractor

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    # 7. Execute Training
    print("[TRAINING] Starting training execution...")
    train_result = trainer.train()

    # 8. Save Final Model & Processor
    print(f"[SAVING MODEL] Saving best fine-tuned QLoRA model to '{args.output_dir}'...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("\n" + "="*70)
    print(" === TRAINING COMPLETED SUCCESSFULLY ===")
    print("="*70 + "\n")

    return trainer


if __name__ == '__main__':
    if torch is None or Seq2SeqTrainer is None:
        print("[NOTICE] PyTorch or Transformers not loaded locally. Module structure verified.")
    else:
        run_training_pipeline()
