import os
import random
from pathlib import Path
from playwright.sync_api import sync_playwright
import base64

try:
    from arabic_reshaper import reshape
    from bidi.algorithm import get_display
    ARABIC_SUPPORT = True
except ImportError:
    ARABIC_SUPPORT = False
    print("For proper Persian text rendering, install these libraries:")
    print("pip install arabic-reshaper python-bidi")


class HTMLDatasetGenerator:
    def __init__(self, config_path='.make-data-env'):
        self.config = self._load_config(config_path)
        self.base_dir = Path(__file__).parent.parent
        self.fonts_dir = self.base_dir / 'fonts'
        self.data_dir = self.base_dir / 'data'
        self.text_file = self.base_dir / 'make-data' / 'persian-text-for-make-data.txt'

        self.persian_text = self._load_persian_text()
        self.words = self.persian_text.split()

        self.persian_digits = '۰۱۲۳۴۵۶۷۸۹'

    def _load_config(self, config_path):
        config = {}
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
        return config

    def _get_config_int(self, key):
        return int(self.config.get(key, 0))

    def _load_persian_text(self):
        with open(self.text_file, 'r', encoding='utf-8') as f:
            return f.read()

    def _get_random_text(self):
        min_len = self._get_config_int('TEXT_LENGTH_MIN')
        max_len = self._get_config_int('TEXT_LENGTH_MAX')
        text_length = random.randint(min_len, max_len)

        start_idx = random.randint(0, len(self.words) - text_length)
        selected_words = self.words[start_idx:start_idx + text_length]
        return ' '.join(selected_words)

    def _get_random_digits(self):
        digit_count = random.randint(3, 10)
        return ''.join(random.choices(self.persian_digits, k=digit_count))

    def _font_to_base64(self, font_path):
        with open(font_path, 'rb') as f:
            font_data = f.read()
        return base64.b64encode(font_data).decode('utf-8')

    def _create_html_template(self, text, font_base64, font_name, font_size):
        font_format = 'truetype'
        if font_name.endswith('.otf'):
            font_format = 'opentype'

        html = f"""
<!DOCTYPE html>
<html dir="rtl" lang="fa">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        @font-face {{
            font-family: 'CustomFont';
            src: url(data:font/{font_format};base64,{font_base64});
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            background-color: rgb(255, 255, 255);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }}

        .text-container {{
            font-family: 'CustomFont', Tahoma, Arial;
            font-size: {font_size}px;
            color: rgb(0, 0, 0);
            text-align: center;
            direction: rtl;
            unicode-bidi: embed;
            line-height: 1.6;
            padding: 20px;
            max-width: 90%;
            word-wrap: break-word;
        }}
    </style>
</head>
<body>
    <div class="text-container">{text}</div>
</body>
</html>
        """
        return html

    def _generate_images_for_font(self, font_path, font_dir, font_name, image_count, is_digit=False):
        font_size_min = self._get_config_int('FONT_SIZE_MIN')
        font_size_max = self._get_config_int('FONT_SIZE_MAX')

        img_width = self._get_config_int('IMAGE_WIDTH_MAX')
        img_height = self._get_config_int('IMAGE_HEIGHT_MAX')

        prefix = "digit" if is_digit else "text"

        font_base64 = self._font_to_base64(font_path)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': img_width, 'height': img_height})

            for img_idx in range(image_count):
                try:
                    text = self._get_random_digits() if is_digit else self._get_random_text()
                    font_size = random.randint(font_size_min, font_size_max)

                    html_content = self._create_html_template(
                        text, font_base64, font_path.name, font_size
                    )

                    page.set_content(html_content)
                    page.wait_for_timeout(500)

                    image_path = font_dir / f"{font_path.stem}_{prefix}_{img_idx:04d}.png"
                    page.screenshot(path=str(image_path), full_page=False)

                    if (img_idx + 1) % 10 == 0:
                        print(f"{prefix}: {img_idx + 1}/{image_count} images")

                except Exception as e:
                    print(f"Error generating {prefix} image number {img_idx}: {e}")
                    continue

            browser.close()

    def generate_dataset(self):
        if not ARABIC_SUPPORT:
            print("Warning: Persian rendering libraries are not installed!")
            print("Run this command for proper rendering:")
            print("pip install arabic-reshaper python-bidi")

        self.data_dir.mkdir(exist_ok=True)

        font_files = list(self.fonts_dir.glob('*.ttf')) + list(self.fonts_dir.glob('*.otf'))

        if not font_files:
            print("No fonts found in the fonts directory!")
            return

        print(f"Number of fonts found: {len(font_files)}")

        img_width = self._get_config_int('IMAGE_WIDTH_MAX')
        img_height = self._get_config_int('IMAGE_HEIGHT_MAX')
        font_size_min = self._get_config_int('FONT_SIZE_MIN')
        font_size_max = self._get_config_int('FONT_SIZE_MAX')

        print(f"Image size: {img_width}x{img_height} pixels")
        print(f"Font size range: {font_size_min}-{font_size_max} pixels")
        print("Color: Black on White")

        images_per_font = self._get_config_int('IMAGES_PER_FONT')
        digit_images_per_font = self._get_config_int('DIGIT_IMAGES_PER_FONT')

        for font_idx, font_path in enumerate(font_files, 1):
            font_name = font_path.stem
            print(f"\n{'=' * 60}")
            print(f"[{font_idx}/{len(font_files)}] Processing font: {font_name}")
            print(f"{'=' * 60}")

            font_dir = self.data_dir / font_name
            font_dir.mkdir(exist_ok=True)

            print("\nGenerating text images...")
            self._generate_images_for_font(font_path, font_dir, font_name, images_per_font, is_digit=False)

            print("\nGenerating digit images...")
            self._generate_images_for_font(font_path, font_dir, font_name, digit_images_per_font, is_digit=True)

            print(f"\nCompleted: {font_name}")
            print(f"- {images_per_font} text images")
            print(f"- {digit_images_per_font} digit images")

        print("\n" + "=" * 60)
        print("Dataset generation completed successfully!")
        print(f"Output path: {self.data_dir}")
        print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("Starting Persian font dataset generation (black & white)")
    print("=" * 60)

    print("\nRequired dependencies:")
    print("pip install playwright arabic-reshaper python-bidi")
    print("playwright install chromium")
    print()

    generator = HTMLDatasetGenerator()
    generator.generate_dataset()