# 🎙️ Neyshekar ASR - Whisper Large-v3 Fine-Tuning Pipeline (QLoRA 4-bit)

پایپ‌لاین کامل، بهینه و تولیدی برای پیش‌پردازش داده‌های صوتی فارسی و آموزش مدل **Whisper Large-v3** با تکنیک **QLoRA (Quantized Low-Rank Adaptation)** و **4-Bit NF4 Quantization**.

---

## 📌 ویژگی‌های کلیدی پروژه

1. **پیش‌پردازش و تمیزکاری داده‌ها (Task 1):**
   - حذف همزادها و نمونه‌های تکراری صوتی و متنی (Exact & Approximate Deduplication).
   - اعتبارسنجی فایل‌های صوتی (بررسی سلامت فایل، تعداد کانال، نرخ نمونه‌برداری ۱۶ کیلوهرتز).
   - فیلترسازی نمونه‌های صوتی بر اساس نرخ گفتار (Speech Rate Filtering بین ۱.۵ تا ۲۲ کاراکتر بر ثانیه).
   - نرمالسازی متنی اختصاصی فارسی (اصلاح نویسه‌های عربی، تبدیل اعداد به حروف، حذف علامت‌های نگارشی).
   - تقسیم‌بندی عادلانه داده‌ها به **۸۵٪ آموزش (Train)** و **۱۵٪ ارزیابی (Validation)**.

2. **معماری آموزش QLoRA (Task 2):**
   - بارگذاری مدل **`openai/whisper-large-v3`** با کوانتایزیشن ۴-بیتی NF4 توسط `bitsandbytes`.
   - تزریق آداپتورهای **LoRA (PEFT)** روی تمام لایه‌های خطی کلیدی (`q_proj`, `v_proj`).
   - ثبت هوک `make_inputs_require_grad` روی لایه `conv1` انکودر جهت تضمین محاسبه گرادیان در Gradient Checkpointing.
   - پاکسازی کامل آرگومان‌های تداخلی `input_ids` در DataCollator و Wrapper پایداری اختصاصی PEFT.
   - محاسبه خودکار معیارهای ارزیابی **WER (Word Error Rate)** و **CER (Character Error Rate)** همراه با نرمالسازی متنی.
   - مدیریت مصرف حافظه VRAM (قابل اجرا روی GPU‌های حداقل 12GB مانند RTX 3060 / T4 / V100 / A100).

---

## 📂 ساختار پروژه (Project Structure)

```text
neyshekar_asr/
├── config.py             # تنظیمات عمومی پروژه (Hyperparameters, Paths, Seed)
├── data_prep.py          # ماژول اصلی اجرای Task 1 (پیش‌پردازش داده‌ها)
├── dataset.py            # ماژول بارگذاری دیتاست و DataCollator برای Whisper
├── model.py              # ساختار بارگذاری مدل QLoRA 4-Bit Whisper
├── metrics.py            # تابع محاسبه معیارهای ارزیابی WER و CER
├── train.py              # اسکریپت اصلی آموزش و مدیریت CLI
├── requirements.txt      # پیش‌نیازها و کتابخانه‌های پایتون
├── README.md             # مستندات و راهنمای اجرای پروژه
├── data/                 # پوشه خروجی داده‌های پردازش‌شده
│   ├── train.csv         # داده‌های آموزش (۸۵٪)
│   └── val.csv           # داده‌های ارزیابی (۱۵٪)
└── src/                  # سورس کدهای داخلی پایپ‌لاین
    ├── config.py
    ├── data_prep.py
    ├── dataset.py
    ├── metrics.py
    ├── model.py
    ├── normalizer.py
    ├── text_cleaner.py
    └── train.py
```

---

## ⚡ راهنمای نصب و آماده‌سازی (Environment Setup)

### ۱. کلون کردن مخزن و نصب پیش‌نیازها

```bash
# clone repository
git clone https://github.com/YOUR_USERNAME/neyshekar_asr.git
cd neyshekar_asr

# create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # In Linux/Mac
# or
.venv\Scripts\activate     # In Windows

# install requirements
pip install -r requirements.txt
```

---

## 🚀 راهنمای اجرا (Execution Guide)

### گام ۱: اجرای پیش‌پردازش داده‌ها (Task 1)

در صورتی که می‌خواهید داده‌های خام صوتی را از ابتدا پاکسازی و آماده کنید:

```bash
python data_prep.py
```
> **خروجی:** فایل‌های `data/train.csv` (۸۵٪ داده‌ها) و `data/val.csv` (۱۵٪ داده‌ها) تولید می‌شوند.

---

### گام ۲: اجرای تست سریع صحت کدهای آموزش (Smoke Test)

قبل از اجرای آموزش کامل، می‌توانید برای اطمینان از صحت تمام ماژول‌ها و عدم وقوع خطای OOM، آموزش را روی **۳۰ گام اول** اجرا کنید:

```bash
python train.py --max_steps 30 --output_dir ./whisper-large-v3-smoke-test
```

---

### گام ۳: اجرای آموزش کامل مدل روی GPU (Full Fine-Tuning)

برای اجرای آموزش کامل روی تمام epochs:

```bash
python train.py --max_steps -1 --epochs 3 --output_dir ./whisper-large-v3-neyshekar-qlora
```

---

## ⚙️ آرگومان‌های خط فرمان (CLI Arguments for `train.py`)

| آرگومان | مقدار پیش‌فرض | توضیحات |
| :--- | :--- | :--- |
| `--max_steps` | `-1` | تعداد گام‌های آموزش (`-1` یعنی آموزش کامل تمام داده‌ها) |
| `--epochs` | `3` | تعداد دوره‌های آموزش (Epochs) |
| `--output_dir` | `./whisper-large-v3-neyshekar-qlora` | مسیر ذخیره‌سازی چک‌پوینت‌ها و مدل نهایی |
| `--train_csv` | `data/train.csv` | مسیر فایل داده‌های آموزش |
| `--val_csv` | `data/val.csv` | مسیر فایل داده‌های ارزیابی |

---

## 📊 معیارهای ارزیابی و گزارش‌گیری (Metrics & TensorBoard)

لاگ‌های آموزش و معیارهای WER و CER در مسیر `./logs` ذخیره می‌شوند. برای مشاهده زنده نمودارها در TensorBoard:

```bash
tensorboard --logdir ./logs
```

---

## 🛡️ عیب‌یابی و پایداری (Troubleshooting & Stability)

- **پیشگیری از خطای OOM:** در صورت محدودیت VRAM، در `config.py` مقادیر `PER_DEVICE_TRAIN_BATCH_SIZE=8` و `GRADIENT_ACCUMULATION_STEPS=4` قرار داده شده‌اند تا حافظه مصرفی کاملاً کنترل شود.
- **سازگاری PEFT با Whisper:** پچ اختصاصی `safe_base_forward` در `model.py` اعمال شده که تمام کلیدهای تداخلی `input_ids` را فیلتر کرده و پایداری کامل آموزش را تضمین می‌کند.

---
**توسعه‌یافته برای سنجش و Fine-tuning مدل‌های گفتار به متن فارسی (Neyshekar ASR).**
