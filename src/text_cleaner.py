import re
try:
    from num2fawords import words as num2fa
except ImportError:
    num2fa = None

# Arabic to Persian character mapping
ARABIC_TO_PERSIAN_MAP = {
    '\u064a': '\u06cc',  # Arabic Yah 'ي' -> Persian Yeh 'ی'
    '\u0643': '\u06a9',  # Arabic Kaf 'ك' -> Persian Kaf 'ک'
    '\u0629': '\u0647',  # Teh Marbuta 'ة' -> Heh 'ه'
    '\u0623': '\u0627',  # Alef with Hamza Above 'أ' -> Alef 'ا'
    '\u0625': '\u0627',  # Alef with Hamza Below 'إ' -> Alef 'ا'
    '\u0671': '\u0627',  # Alef Wasla 'ٱ' -> Alef 'ا'
    '\u0624': '\u0648',  # Waw with Hamza 'ؤ' -> Waw 'و'
    '\u0626': '\u06cc',  # Yeh with Hamza 'ئ' -> Yeh 'ی'
    '\u0649': '\u06cc',  # Alef Maksura 'ى' -> Yeh 'ی'
}

ARABIC_CHARS_PATTERN = re.compile(r'[\u064a\u0643\u0629\u0623\u0625\u0671\u0624\u0626\u0649]')

# Diacritics / Harakat regex (fatha, kasra, damma, tanwin, tashdeed, sukun, etc.)
DIACRITICS_PATTERN = re.compile(r'[\u064B-\u0652\u0670\u06D6-\u06ED]')

# Digits pattern (English, Persian, Arabic digits)
ENGLISH_DIGITS = "0123456789"
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS  = "٠١٢٣٤٥٦٧٨٩"

DIGIT_TO_ENGLISH = str.maketrans(
    PERSIAN_DIGITS + ARABIC_DIGITS,
    ENGLISH_DIGITS + ENGLISH_DIGITS
)

DIGITS_PATTERN = re.compile(r'[\d۰-۹٠-٩]+')

def has_arabic_chars(text: str) -> bool:
    """Checks if text contains Arabic-specific characters needing conversion to Persian."""
    if not isinstance(text, str):
        return False
    return bool(ARABIC_CHARS_PATTERN.search(text))

def convert_arabic_to_persian(text: str) -> str:
    """Converts Arabic characters (ي, ك, ة, أ, إ, etc.) to standard Persian equivalents."""
    if not isinstance(text, str):
        return ""
    for ar_char, fa_char in ARABIC_TO_PERSIAN_MAP.items():
        text = text.replace(ar_char, fa_char)
    return text

def has_diacritics(text: str) -> bool:
    """Checks if text contains Arabic/Persian diacritics (Harakat)."""
    if not isinstance(text, str):
        return False
    return bool(DIACRITICS_PATTERN.search(text))

def remove_diacritics(text: str) -> str:
    """Removes Persian/Arabic diacritics (fatha, kasra, damma, tanwin, tashdeed, sukun, etc.)."""
    if not isinstance(text, str):
        return ""
    return DIACRITICS_PATTERN.sub('', text)

def has_digits(text: str) -> bool:
    """Checks if text contains any numeric digits (English, Persian, or Arabic)."""
    if not isinstance(text, str):
        return False
    return bool(DIGITS_PATTERN.search(text))

def lexicalize_numbers(text: str) -> str:
    """Converts digits (e.g., '۱۲۵', '125') into Persian spoken words (e.g., 'صد و بیست و پنج')."""
    if not isinstance(text, str):
        return ""
    
    def _repl(match):
        raw_num_str = match.group(0)
        ascii_num_str = raw_num_str.translate(DIGIT_TO_ENGLISH)
        try:
            val = int(ascii_num_str)
            if num2fa is not None:
                return f" {num2fa(val)} "
            return raw_num_str
        except (ValueError, OverflowError):
            return raw_num_str

    return DIGITS_PATTERN.sub(_repl, text)

def remove_spaces_and_zwnj(text: str) -> str:
    """Removes spaces, tabs, newlines, zero-width non-joiners (ZWNJ), and punctuation for strict duplicate matching."""
    if not isinstance(text, str):
        return ""
    # Remove punctuation & non-alphanumeric (keep Persian characters & numbers)
    text = re.sub(r'[^\w]', '', text)
    # Remove whitespace and ZWNJ (\u200c)
    text = re.sub(r'[\s\u200c]', '', text)
    return text

def apply_step1_normalization(text: str) -> str:
    """Applies Step 1 normalization: Arabic->Persian character conversion and number lexicalization."""
    if not isinstance(text, str):
        return ""
    text = convert_arabic_to_persian(text)
    text = lexicalize_numbers(text)
    return text
