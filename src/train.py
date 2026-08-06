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
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help=(
            "Override evaluation frequency. If not set: uses config.EVAL_STEPS for full runs, "
            "but is auto-scaled down for smoke tests (--max_steps > 0) so that at least one "
            "eval/generate/checkpoint cycle actually runs during the smoke test. This matters "
            "because config.EVAL_STEPS=100 would never trigger during e.g. --max_steps 30, "
            "silently skipping the eval + WER/CER + save-best-checkpoint code path."
        )
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Override checkpoint-save frequency. Same auto-scaling behavior as --eval_steps."
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=config.MAX_EVAL_SAMPLES,
        help=(
            "Number of validation rows used for PERIODIC in-training evaluation. Each eval runs "
            "autoregressive generate(), so the full 5,900-row split costs hours across a 3-epoch "
            "run. Pass -1 to always evaluate on the full split."
        )
    )
    parser.add_argument(
        "--max_shards",
        type=int,
        default=None,
        help=(
            "Download only the first N of the 15 parquet shards (~485 MB each) instead of the full "
            "~7.3 GB. Intended for Colab smoke tests, where downloading everything just to keep "
            "1,000 samples is wasteful. NEVER use this for the real 3-epoch run or reported metrics."
        )
    )
    # ---- throughput knobs (see the efficiency ablation in the README) ----
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        help=(
            "Disable gradient checkpointing. Trades VRAM for ~1.3-1.5x faster training with "
            "IDENTICAL gradients, so model quality is unaffected. Needs more memory: safe on a "
            "24 GB L4, likely OOM on a 16 GB T4."
        )
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        help="Per-device train batch size. Keep train_batch_size * grad_accum == 16 to preserve "
             "the validated effective batch size and learning-rate recipe."
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=config.GRADIENT_ACCUMULATION_STEPS,
        help="Gradient accumulation steps. See --train_batch_size."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=config.PER_DEVICE_EVAL_BATCH_SIZE,
        help="Per-device eval batch size. Affects evaluation SPEED ONLY -- it cannot change "
             "WER/CER, so raise it as far as VRAM allows."
    )
    parser.add_argument(
        "--dataloader_workers",
        type=int,
        default=config.DATALOADER_NUM_WORKERS,
        help="Worker processes for the dataloader (0 = load in the main process)."
    )
    parser.add_argument(
        "--no_quantization",
        action="store_true",
        help=(
            "Load the base model in bf16 instead of 4-bit NF4. Removes the per-step "
            "dequantization overhead and lets LoRA adapt EXACT rather than approximated "
            "weights. Needs ~2.2 GB more VRAM; safe on a 24 GB L4."
        )
    )
    parser.add_argument(
        "--no_vad_trim",
        action="store_true",
        help="Disable trimming of leading/trailing silence before feature extraction."
    )
    parser.add_argument(
        "--no_peak_norm",
        action="store_true",
        help="Disable peak normalisation of the waveform before feature extraction."
    )
    parser.add_argument(
        "--final_full_eval",
        action="store_true",
        help=(
            "After training, run one final evaluation on the FULL validation set and save the "
            "result to final_eval_metrics.json. Use this for the numbers you report, so the "
            "headline WER/CER are not based on a subset."
        )
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
    is_smoke_test = args.max_steps is not None and args.max_steps > 0
    max_samples = 1000 if is_smoke_test else None
    train_dataset, val_dataset, processor, data_collator = get_datasets_and_collator(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        max_samples=max_samples,
        max_shards=args.max_shards,
        enable_vad_trim=config.ENABLE_VAD_TRIM and not args.no_vad_trim,
        enable_peak_norm=config.ENABLE_PEAK_NORM and not args.no_peak_norm
    )

    # Guard against accidentally producing "final" numbers from a partial download.
    if args.max_shards and not is_smoke_test:
        print("\n[WARNING] --max_shards is set on a FULL training run. Only part of the corpus was "
              "downloaded, so these results are NOT reportable. Remove --max_shards for the real run.\n")

    # 2b. Resolve eval/save step frequency.
    #     IMPORTANT: config.EVAL_STEPS/SAVE_STEPS default to 100, which is larger than a typical
    #     smoke-test run (e.g. --max_steps 30). If left as-is, the smoke test would finish without
    #     ever triggering an eval/generate/WER-CER/checkpoint-save cycle -- exactly the code paths
    #     most likely to hide a bug (predict_with_generate, compute_metrics, best-model selection).
    #     So unless explicitly overridden, we auto-scale eval/save steps down during smoke tests.
    if args.eval_steps is not None:
        eval_steps = args.eval_steps
    elif is_smoke_test:
        eval_steps = max(1, args.max_steps // 3)
    else:
        eval_steps = config.EVAL_STEPS

    if args.save_steps is not None:
        save_steps = args.save_steps
    else:
        # load_best_model_at_end requires save_steps to be a round multiple of eval_steps
        save_steps = eval_steps if is_smoke_test else config.SAVE_STEPS

    if is_smoke_test:
        print(f"[SMOKE TEST] Auto-scaled eval_steps={eval_steps}, save_steps={save_steps} "
              f"(max_steps={args.max_steps}) so the eval/WER/CER/checkpoint path is actually "
              f"exercised, not just the training loss path.")

    # 2c. Optionally subsample the validation set used for PERIODIC evaluation.
    #     Kept separate from the full split so the final reported metrics can still be computed
    #     over everything (see --final_full_eval below).
    full_val_dataset = val_dataset
    if args.max_eval_samples is not None and args.max_eval_samples > 0 \
            and len(val_dataset) > args.max_eval_samples:
        # Shuffle with the global seed before selecting, so the subset is a representative random
        # sample rather than the first N rows, yet is identical on every run (reproducibility).
        val_dataset = val_dataset.shuffle(seed=config.SEED).select(range(args.max_eval_samples))
        print(f"[EVAL SUBSET] Periodic evaluation will use {len(val_dataset):,} of "
              f"{len(full_val_dataset):,} validation rows (deterministic, seed={config.SEED}). "
              f"Use --final_full_eval to report final metrics on the full split.")

    # 2d. Resolve throughput settings and guard the effective batch size.
    #     Effective batch = train_batch_size * grad_accum. The learning-rate recipe (1e-3 with
    #     50 warmup steps) was validated at effective batch 16; changing that number changes the
    #     optimisation dynamics, so redistributing 16 across batch/accumulation is safe but
    #     changing the product is not. Warn loudly rather than silently altering the recipe.
    use_gradient_checkpointing = config.GRADIENT_CHECKPOINTING and not args.no_gradient_checkpointing
    effective_batch = args.train_batch_size * args.grad_accum
    print(f"[THROUGHPUT] train_batch={args.train_batch_size} x grad_accum={args.grad_accum} "
          f"-> effective batch = {effective_batch} | eval_batch={args.eval_batch_size} | "
          f"dataloader_workers={args.dataloader_workers}")
    if effective_batch != config.PER_DEVICE_TRAIN_BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS:
        print(f"[THROUGHPUT][WARNING] Effective batch is {effective_batch}, not "
              f"{config.PER_DEVICE_TRAIN_BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}. This "
              f"CHANGES the optimisation recipe -- the learning rate ({config.LEARNING_RATE}) was "
              f"tuned for the original value and should be rescaled. Results are not directly "
              f"comparable to the baseline configuration.")

    # 2e. Precision must agree between the model weights and the Trainer's autocast setting.
    #     With quantization off we load bf16 weights, so the Trainer must use bf16 too --
    #     leaving fp16=True there would mix precisions and can silently destabilise training.
    #     transformers also rejects fp16 and bf16 being enabled at the same time.
    use_quantization = config.USE_QUANTIZATION and not args.no_quantization
    if use_quantization:
        use_fp16, use_bf16 = config.FP16, False
    else:
        # Detect bf16 via compute capability rather than torch.cuda.is_bf16_supported():
        # that helper can report False when CUDA has not been initialised yet, which silently
        # downgraded an Ada-class L4 (SM 8.9) to fp16. bf16 needs SM 8.0+ (Ampere and newer).
        bf16_capable = False
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            bf16_capable = major >= 8
            print(f"[PRECISION] GPU {torch.cuda.get_device_name(0)} "
                  f"(compute capability {major}.{minor}) -> bf16 {'supported' if bf16_capable else 'NOT supported'}")
        want_bf16 = config.USE_BF16_WHEN_UNQUANTIZED and bf16_capable
        use_fp16, use_bf16 = (False, True) if want_bf16 else (True, False)
    print(f"[PRECISION] quantization={'4-bit NF4' if use_quantization else 'OFF'} | "
          f"fp16={use_fp16} | bf16={use_bf16}")

    # 3. Load Model (quantized 4-bit, or full-precision bf16)
    model = get_whisper_qlora_model(
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_quantization=use_quantization
    )

    # 4. Prepare Compute Metrics Function
    compute_metrics_fn = get_compute_metrics_fn(processor=processor)

    # 5. Define Seq2Seq Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        dataloader_num_workers=args.dataloader_workers,
        learning_rate=config.LEARNING_RATE,
        warmup_steps=config.WARMUP_STEPS,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=use_gradient_checkpointing,
        predict_with_generate=True,
        generation_max_length=225,
        eval_strategy=config.EVAL_STRATEGY,
        eval_steps=eval_steps,
        save_steps=save_steps,
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

    # 9. Optional final evaluation on the FULL validation split.
    #    Periodic evals may have run on a subset for speed; these are the numbers worth reporting.
    if args.final_full_eval and len(full_val_dataset) > len(val_dataset):
        print(f"\n[FINAL EVAL] Evaluating best model on the FULL validation set "
              f"({len(full_val_dataset):,} rows). This runs generate() over every row and will "
              f"take a while...")
        final_metrics = trainer.evaluate(eval_dataset=full_val_dataset, metric_key_prefix="final")
        trainer.log_metrics("final", final_metrics)
        trainer.save_metrics("final", final_metrics)
        print(f"[FINAL EVAL] Full-validation results: "
              f"WER={final_metrics.get('final_wer')}%, CER={final_metrics.get('final_cer')}%")

    print("\n" + "="*70)
    print(" === TRAINING COMPLETED SUCCESSFULLY ===")
    print("="*70 + "\n")

    return trainer


if __name__ == '__main__':
    if torch is None or Seq2SeqTrainer is None:
        print("[NOTICE] PyTorch or Transformers not loaded locally. Module structure verified.")
    else:
        run_training_pipeline()
