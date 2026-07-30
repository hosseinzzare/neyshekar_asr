import re

try:
    from num2fawords import words as num2fa
except ImportError:
    num2fa = None

class PersianTextNormalizer:
    """
    Persian text normalizer for ASR tasks.
    Standardizes Arabic characters to Persian, converts digits to Persian words,
    and removes diacritics and unnecessary punctuation.
    """
    def __init__(self, lexicalize_numbers: bool = True):
        self.lexicalize_numbers = lexicalize_numbers
        
        # Arabic to Persian character mapping
        self.arabic_to_persian_map = {
            'ي': 'ی',
            'ك': 'ک',
            'ة': 'ه',
            'ٱ': 'ا',
            'إ': 'ا',
            'أ': 'ا',
            'ئ': 'ی',
            'ؤ': 'و',
            'ى': 'ی',
        }
        
        # Digits mapping (English and Arabic -> Persian digits)
        self.digits_map = str.maketrans("0123456789٠١٢٣٤٥٦٧٨٩", "۰۱۲۳۴۵۶۷۸۹۰۱۲۳۴۵۶۷۸۹")
        self.english_digits = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

    def remove_diacritics(self, text: str) -> str:
        """Removes Persian/Arabic diacritics (fatha, kasra, damma, tanwin, tashdeed, sukun)."""
        paddings = re.compile(r'[\u064B-\u0652\u0670]')
        return paddings.sub('', text)

    def normalize_chars(self, text: str) -> str:
        """Converts Arabic letters and digits to Persian equivalents."""
        for ar, fa in self.arabic_to_persian_map.items():
            text = text.replace(ar, fa)
        return text.translate(self.digits_map)

    def lexicalize_digits(self, text: str) -> str:
        """Converts numbers (e.g. '۱۲۵' or '125') to Persian words ('صد و بیست و پنج')."""
        if not self.lexicalize_numbers or num2fa is None:
            return text

        def _repl(match):
            num_str = match.group(0).translate(self.english_digits)
            try:
                val = int(num_str)
                return f" {num2fa(val)} "
            except Exception:
                return match.group(0)

        # Match sequences of digits (Persian, Arabic, English)
        text = re.sub(r'[\d۰-۹٠-٩]+', _repl, text)
        return text

    def remove_punctuation(self, text: str) -> str:
        """Removes punctuation and keeps Persian alphanumeric characters, spaces, and ZWNJ."""
        text = re.sub(r'[^\w\s\u200c]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def remove_spaces_and_zwnj(self, text: str) -> str:
        """Removes all whitespace and zero-width non-joiner for condensed matching."""
        return re.sub(r'[\s\u200c]', '', text)

    def normalize(self, text: str) -> str:
        """Full normalization pipeline."""
        if not isinstance(text, str):
            return ""
        text = self.remove_diacritics(text)
        text = self.normalize_chars(text)
        text = self.lexicalize_digits(text)
        text = self.remove_punctuation(text)
        return text

