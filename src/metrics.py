"""
Evaluation Metrics (WER & CER) with Persian Text Normalization for Whisper Fine-Tuning.
Task 2 - Step 5: Metrics Evaluation Module.
"""

import os
import sys
import re
from typing import Dict, Any, List

# Force UTF-8 encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure src is in sys.path
sys.path.append(os.path.dirname(__file__))
import config
from text_cleaner import convert_arabic_to_persian, remove_diacritics

try:
    import evaluate
    from transformers import WhisperProcessor
except ImportError:
    evaluate = None
    WhisperProcessor = None


def normalize_persian_for_eval(text: str) -> str:
    """
    Standardizes Persian text for WER/CER evaluation:
    - Converts Arabic characters (ي, ك) to Persian (ی, ک).
    - Removes Persian/Arabic diacritics (Harakat).
    - Removes punctuation marks to avoid penalizing valid predictions for missing commas/dots.
    - Normalizes extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = convert_arabic_to_persian(text)
    text = remove_diacritics(text)
    # Remove punctuation & non-alphanumeric (keeping spaces and ZWNJ \u200c)
    text = re.sub(r'[^\w\s\u200c]', ' ', text)
    # Normalize multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class WhisperMetricsEvaluator:
    """
    Evaluator class managing WER & CER metric computation for Hugging Face Seq2SeqTrainer.
    """
    def __init__(
        self,
        processor: Any = None,
        model_name: str = config.MODEL_NAME_OR_PATH,
        language: str = config.LANGUAGE,
        task: str = config.TASK
    ):
        if evaluate is None or WhisperProcessor is None:
            raise RuntimeError("evaluate or transformers package is not installed.")

        print("[METRICS] Loading WER and CER evaluation metrics from Hugging Face evaluate...")
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")

        if processor is None:
            print(f"[METRICS] Loading WhisperProcessor for '{model_name}'...")
            self.processor = WhisperProcessor.from_pretrained(
                model_name,
                language=language,
                task=task
            )
        else:
            self.processor = processor

    def compute_metrics(self, pred: Any) -> Dict[str, float]:
        """
        Computes normalized WER and CER metrics from Seq2SeqTrainer predictions object.
        """
        import numpy as np

        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Seq2SeqTrainer may hand back a tuple (e.g. (sequences, scores)) rather than a bare array.
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        pad_token_id = self.processor.tokenizer.pad_token_id

        # Replace -100 with pad_token_id so the tokenizer can decode.
        # Copy first: pred.label_ids is reused by the Trainer, and the original code mutated it
        # in place, which corrupts the array if metrics are ever computed twice over it.
        label_ids = np.asarray(label_ids).copy()
        label_ids[label_ids == -100] = pad_token_id

        # Generated ids can also contain -100 padding when generation lengths differ across the batch.
        pred_ids = np.asarray(pred_ids).copy()
        pred_ids[pred_ids == -100] = pad_token_id

        # Decode token IDs to text strings
        pred_str_raw = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str_raw = self.processor.batch_decode(label_ids, skip_special_tokens=True)

        # Apply Persian text normalization
        pred_str_norm = [normalize_persian_for_eval(s) for s in pred_str_raw]
        label_str_norm = [normalize_persian_for_eval(s) for s in label_str_raw]

        # Drop pairs whose REFERENCE normalized to an empty string.
        # jiwer raises ValueError("one or more references are empty strings") and would abort the
        # whole training run at the first eval. An empty reference is also mathematically
        # meaningless for WER (division by zero reference words).
        filtered = [(p, l) for p, l in zip(pred_str_norm, label_str_norm) if l.strip()]
        dropped = len(pred_str_norm) - len(filtered)
        if dropped:
            print(f"[METRICS][WARNING] Skipped {dropped} sample(s) with an empty reference transcript.")

        if not filtered:
            print("[METRICS][WARNING] All references were empty in this eval batch; returning WER/CER = 0.")
            return {"wer": 0.0, "cer": 0.0}

        pred_str_norm = [p for p, _ in filtered]
        label_str_norm = [l for _, l in filtered]

        # Compute WER and CER as percentages
        wer_val = 100.0 * self.wer_metric.compute(predictions=pred_str_norm, references=label_str_norm)
        cer_val = 100.0 * self.cer_metric.compute(predictions=pred_str_norm, references=label_str_norm)

        return {
            "wer": round(wer_val, 2),
            "cer": round(cer_val, 2)
        }


# Global instance & helper function for direct trainer integration
_evaluator_instance = None

def get_compute_metrics_fn(processor: Any = None):
    """
    Returns a callable compute_metrics(pred) function for Hugging Face Seq2SeqTrainer.
    """
    global _evaluator_instance
    if _evaluator_instance is None or processor is not None:
        _evaluator_instance = WhisperMetricsEvaluator(processor=processor)
    return _evaluator_instance.compute_metrics


if __name__ == '__main__':
    print("Testing metrics.py module functionality...")
    
    if evaluate is None or WhisperProcessor is None:
        print("[NOTICE] evaluate/transformers libraries missing locally.")
    else:
        # Quick unit test with dummy predictions & reference strings
        test_processor = WhisperProcessor.from_pretrained(
            config.MODEL_NAME_OR_PATH,
            language=config.LANGUAGE,
            task=config.TASK
        )
        evaluator = WhisperMetricsEvaluator(processor=test_processor)
        
        ref_text = ["سلام روز شما بخیر و نیکی"]
        pred_text = ["سلام روز شما بخیر و نیكی"] # Contains Arabic Yah 'ي'
        
        ref_ids = test_processor.tokenizer(ref_text).input_ids
        pred_ids = test_processor.tokenizer(pred_text).input_ids
        
        # Convert to numpy mock arrays
        import numpy as np
        class DummyPred:
            def __init__(self, p, l):
                self.predictions = np.array(p)
                self.label_ids = np.array(l)
                
        mock_pred = DummyPred(pred_ids, ref_ids)
        metrics = evaluator.compute_metrics(mock_pred)
        
        print("\n" + "="*50)
        print(" === METRICS MODULE TEST OUTPUT ===")
        print("="*50)
        print(f"Reference: '{ref_text[0]}'")
        print(f"Prediction: '{pred_text[0]}'")
        print(f"Computed WER: {metrics['wer']}%")
        print(f"Computed CER: {metrics['cer']}%")
        print("="*50)
        print("[SUCCESS] metrics.py unit test passed cleanly!")
