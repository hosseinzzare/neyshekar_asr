---
license: apache-2.0
language:
- fa
base_model: openai/whisper-large-v3
datasets:
- shekar-ai/neyshekar-v4-persian-asr-fa
tags:
- automatic-speech-recognition
- persian
- farsi
- whisper
- qlora
- peft
library_name: peft
pipeline_tag: automatic-speech-recognition
metrics:
- wer
- cer
model-index:
- name: neyshekar-whisper-large-v3-lora
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Neyshekar v4 Persian ASR
      type: shekar-ai/neyshekar-v4-persian-asr-fa
      split: validation
    metrics:
    - type: wer
      value: 8.05
      name: WER
    - type: cer
      value: 2.00
      name: CER
---

# Whisper large-v3 fine-tuned for Persian (QLoRA adapter)

A LoRA adapter for `openai/whisper-large-v3`, trained on the Neyshekar Persian speech corpus.
**WER 8.05%, CER 2.00%** over the full 5,900-row validation split.

Code, the five analysis documents and a live demo:
**[github.com/hosseinzzare/neyshekar_asr](https://github.com/hosseinzzare/neyshekar_asr)**

## What the fine-tuning bought

A number on its own says nothing, so the same 200 validation utterances were transcribed twice:
once by the untouched base model, once with this adapter. Identical 4-bit load path, identical
preprocessing, identical generation settings — the adapter is the only variable.

| Scored over the same 200 utterances | Base model | With this adapter | Change |
|---|---:|---:|---:|
| Corpus WER, normalised | 36.97% | 8.74% | −28.2 points |
| Corpus CER, normalised | 8.80% | 2.14% | −6.7 points |
| Transcribed exactly | 23 (11.5%) | 115 (57.5%) | 5× more |
| Predictions containing a digit | 4 | 0 | eliminated |
| Predictions using a ZWNJ | 0 | 116 | learned |

The last two rows matter more than the WER. Every number in the training labels was written out
as words, so a digit in the output is a convention the model has not absorbed; after fine-tuning
there are none. The zero-width non-joiner is the mirror image: the base model never emits one,
this adapter emits 116 against 118 in the references. It learned a Persian orthographic
convention that WER barely registers.

The per-utterance CSVs behind every figure above are in `run_artifacts/` in this repository, so
none of it has to be taken on trust.

## Usage

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import torch, librosa

BASE = "openai/whisper-large-v3"
processor = WhisperProcessor.from_pretrained(BASE, language="persian", task="transcribe")
model = PeftModel.from_pretrained(
    WhisperForConditionalGeneration.from_pretrained(BASE, dtype=torch.float16, device_map="auto"),
    "hosseinzr/neyshekar-whisper-large-v3-lora",
).eval()
model.generation_config.language = "persian"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

audio, _ = librosa.load("clip.wav", sr=16000)
feats = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
feats = feats.to(model.device).half()
with torch.no_grad():
    ids = model.generate(input_features=feats, max_new_tokens=225)
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```

Or run the packaged HTTP service without installing anything:

```bash
git clone https://github.com/hosseinzzare/neyshekar_asr && cd neyshekar_asr
docker build -t neyshekar-asr .
docker run -p 8000:8000 -v neyshekar-cache:/cache neyshekar-asr
curl -F "file=@clip.wav" http://localhost:8000/transcribe
```

The service applies the same silence trimming and peak normalisation used in training, imported
from the training module rather than reimplemented, so it cannot drift from the pipeline these
metrics were measured on.

## Training

| | |
|---|---|
| Method | QLoRA — 4-bit NF4 base, double quantisation, fp16 compute |
| LoRA | `r=32`, `alpha=64`, `dropout=0.05`, on `q_proj` and `v_proj` |
| Trainable | 15,728,640 of 1,559,219,200 parameters — **1.01%** |
| Data | 33,432 train / 5,900 validation, cleaned from 40,008 raw rows |
| Schedule | 3 epochs, 6,270 optimizer steps, effective batch 16 (8 × 2) |
| Learning rate | 1e-3, 50 warmup steps, linear decay |
| Hardware | one NVIDIA L4, 22.03 GiB — **peak use 5.51 GiB** |
| Wall clock | 14 h 19 min |
| Selection | best checkpoint by WER, not by loss |

Audio preprocessing: leading and trailing silence trimmed before feature extraction (pauses
*inside* an utterance are natural speech and are kept), every clip peak-normalised to −3 dBFS.
Both were chosen from measurements on the corpus, not by default.

## Limitations

- **Validation overlaps training by transcript.** 51.9% of validation transcripts also appear in
  the training split with different audio. The split is disjoint by recording but not by
  sentence, so validation loss in particular is optimistic — teacher forcing on a sentence the
  decoder has already learned is close to trivial. Treat 8.05% as an in-domain figure.
- **A quarter of the remaining errors are spelling conventions, not mishearings.** Hand analysis
  of 156 error events found 25.6% are Persian compound spacing and homophone letters — the model
  heard the word correctly and wrote it by a different rule. Rescoring with the zero-width
  non-joiner normalised moves WER by 0.53 points.
- **Colloquial speech drifts toward formal written Persian.** The adapter touches only the query
  and value projections, leaving most of the decoder's language model untouched.
- **Clipped audio was not repaired.** 22.10% of the corpus peaks at the digital ceiling; that
  information is gone and normalisation cannot restore it.
- **The exact training environment was not captured** with `pip freeze` before the machine was
  released. The log establishes transformers 5.x; finer detail than that is not recoverable.

## Files

| | |
|---|---|
| `run_artifacts/error_analysis.csv` | per-utterance predictions and scores, this adapter |
| `run_artifacts/error_analysis_zeroshot.csv` | the same 200 utterances, base model only |
| `run_artifacts/error_categories.csv` | word-level error events with their categories |
| `run_artifacts/train.log` | the full training log, 627 loss points and 5 evaluations |
| `run_artifacts/ablation_logs.tar.gz` | the two runs that OOMed without gradient checkpointing |
