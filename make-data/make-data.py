import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np

try:
    from arabic_reshaper import reshape
    from bidi.algorithm import get_display

    ARABIC_SUPPORT = True
except ImportError:
    ARABIC_SUPPORT = False
    print("⚠️  To properly render Persian text, install the following libraries:")
    print("   pip install arabic-reshaper python-bidi")


class DatasetGenerator:
    def __init__(self, config_path='.make-data-env'):
        """Read configuration from env file"""
        self.config = self._load_config(config_path)
        self.base_dir = Path(__file__).parent.parent
        self.fonts_dir = self.base_dir / 'fonts'
        self.data_dir = self.base_dir / 'data'
        self.text_file = self.base_dir / 'make-data' / 'persian-text-for-make-data.txt'

        # Load Persian text
        self.persian_text = self._load_persian_text()
        self.words = self.persian_text.split()

        # Persian digits
        self.persian_digits = '۰۱۲۳۴۵۶۷۸۹'

    def _load_config(self, config_path):
        """Load configuration file"""
        config = {}
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
        return config

    def _get_config_int(self, key):
        """Get integer value from config"""
        return int(self.config.get(key, 0))

    def _get_config_float(self, key):
        """Get float value from config"""
        return float(self.config.get(key, 0))

    def _load_persian_text(self):
        """Load Persian text from file"""
        with open(self.text_file, 'r', encoding='utf-8') as f:
            return f.read()

    def _parse_colors(self, color_string):
        """Convert color string to list of RGB tuples"""
        colors = []
        for color in color_string.split(';'):
            r, g, b = map(int, color.split(','))
            colors.append((r, g, b))
        return colors

    def _reshape_persian_text(self, text):
        """Reshape Persian text for proper rendering"""
        if ARABIC_SUPPORT:
            reshaped_text = reshape(text)
            bidi_text = get_display(reshaped_text)
            return bidi_text
        else:
            return text

    def _get_random_text(self):
        """Select a random segment of the text"""
        min_len = self._get_config_int('TEXT_LENGTH_MIN')
        max_len = self._get_config_int('TEXT_LENGTH_MAX')
        text_length = random.randint(min_len, max_len)

        start_idx = random.randint(0, len(self.words) - text_length)
        selected_words = self.words[start_idx:start_idx + text_length]

        text = ' '.join(selected_words)
        return self._reshape_persian_text(text)

    def _get_random_digits(self):
        """Select random Persian digits"""
        digit_count = random.randint(3, 10)
        digits = ''.join(random.choices(self.persian_digits, k=digit_count))
        return self._reshape_persian_text(digits)

    def _create_base_image(self, text, font_path, font_size, text_color, bg_color):
        """Create base image with text"""
        try:
            font = ImageFont.truetype(str(font_path), font_size)
        except Exception as e:
            print(f"Error loading font: {font_path} - {e}")
            return None

        # Calculate text size
        temp_img = Image.new('RGB', (1, 1), bg_color)
        draw = ImageDraw.Draw(temp_img)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Create image with padding
        padding = 20
        img_width = text_width + 2 * padding
        img_height = text_height + 2 * padding

        # Clamp image size to configured limits
        min_width = self._get_config_int('IMAGE_WIDTH_MIN')
        max_width = self._get_config_int('IMAGE_WIDTH_MAX')
        min_height = self._get_config_int('IMAGE_HEIGHT_MIN')
        max_height = self._get_config_int('IMAGE_HEIGHT_MAX')

        img_width = max(min_width, min(img_width, max_width))
        img_height = max(min_height, min(img_height, max_height))

        # Create image
        image = Image.new('RGB', (img_width, img_height), bg_color)
        draw = ImageDraw.Draw(image)

        # Draw text centered
        x = (img_width - text_width) // 2
        y = (img_height - text_height) // 2
        draw.text((x, y), text, font=font, fill=text_color)

        return image

    def _apply_blur(self, image):
        """Apply blur effect"""
        blur_min = self._get_config_int('BLUR_MIN')
        blur_max = self._get_config_int('BLUR_MAX')
        blur_radius = random.uniform(blur_min, blur_max)
        return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    def _apply_noise(self, image):
        """Apply noise"""
        noise_min = self._get_config_int('NOISE_MIN')
        noise_max = self._get_config_int('NOISE_MAX')
        noise_level = random.randint(noise_min, noise_max)

        img_array = np.array(image)
        noise = np.random.randint(-noise_level, noise_level, img_array.shape, dtype=np.int16)
        noisy_img = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy_img)

    def _apply_rotation(self, image):
        """Apply rotation"""
        rotation_min = self._get_config_int('ROTATION_MIN')
        rotation_max = self._get_config_int('ROTATION_MAX')
        angle = random.uniform(rotation_min, rotation_max)
        return image.rotate(angle, expand=True, fillcolor=(255, 255, 255))

    def _apply_brightness(self, image):
        """Adjust brightness"""
        brightness_min = self._get_config_float('BRIGHTNESS_MIN')
        brightness_max = self._get_config_float('BRIGHTNESS_MAX')
        factor = random.uniform(brightness_min, brightness_max)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def _apply_grayscale(self, image):
        """Convert image to grayscale"""
        return image.convert('L').convert('RGB')

    def _apply_random_effects(self, image):
        """Apply random effects"""
        blur_prob = self._get_config_float('BLUR_PROBABILITY')
        noise_prob = self._get_config_float('NOISE_PROBABILITY')
        rotation_prob = self._get_config_float('ROTATION_PROBABILITY')
        brightness_prob = self._get_config_float('BRIGHTNESS_PROBABILITY')
        grayscale_prob = self._get_config_float('GRAYSCALE_PROBABILITY')

        if random.random() < blur_prob:
            image = self._apply_blur(image)

        if random.random() < noise_prob:
            image = self._apply_noise(image)

        if random.random() < rotation_prob:
            image = self._apply_rotation(image)

        if random.random() < brightness_prob:
            image = self._apply_brightness(image)

        if random.random() < grayscale_prob:
            image = self._apply_grayscale(image)

        return image

    def _generate_images_for_font(self, font_path, font_dir, font_name, image_count, is_digit=False):
        """Generate images for a single font"""
        text_colors = self._parse_colors(self.config.get('TEXT_COLORS'))
        bg_colors = self._parse_colors(self.config.get('BACKGROUND_COLORS'))
        font_size_min = self._get_config_int('FONT_SIZE_MIN')
        font_size_max = self._get_config_int('FONT_SIZE_MAX')

        prefix = "digit" if is_digit else "text"

        for img_idx in range(image_count):
            try:
                # Select random parameters
                if is_digit:
                    text = self._get_random_digits()
                else:
                    text = self._get_random_text()

                font_size = random.randint(font_size_min, font_size_max)
                text_color = random.choice(text_colors)
                bg_color = random.choice(bg_colors)

                # Create base image
                image = self._create_base_image(text, font_path, font_size, text_color, bg_color)

                if image is None:
                    continue

                # Apply random effects
                image = self._apply_random_effects(image)

                # Save image
                image_path = font_dir / f"{font_name}_{prefix}_{img_idx:04d}.png"
                image.save(image_path)

                if (img_idx + 1) % 10 == 0:
                    print(f"  {prefix}: {img_idx + 1}/{image_count} images")

            except Exception as e:
                print(f"  Error generating {prefix} image {img_idx}: {e}")
                continue

    def generate_dataset(self):
        """Generate full dataset"""
        # Check Persian support
        if not ARABIC_SUPPORT:
            print("\n⚠️  Warning: Persian libraries are not installed!")
            print("To render Persian text correctly, run:")
            print("pip install arabic-reshaper python-bidi\n")

        # Create data directory
        self.data_dir.mkdir(exist_ok=True)

        # Get font list
        font_files = list(self.fonts_dir.glob('*.ttf')) + list(self.fonts_dir.glob('*.otf'))

        if not font_files:
            print("❌ No fonts found in fonts directory!")
            return

        print(f"📝 Number of fonts found: {len(font_files)}")

        images_per_font = self._get_config_int('IMAGES_PER_FONT')
        digit_images_per_font = self._get_config_int('DIGIT_IMAGES_PER_FONT')

        # Process each font
        for font_idx, font_path in enumerate(font_files, 1):
            font_name = font_path.stem
            print(f"\n{'=' * 60}")
            print(f"[{font_idx}/{len(font_files)}] 🔤 Processing font: {font_name}")
            print(f"{'=' * 60}")

            # Create directory for this font
            font_dir = self.data_dir / font_name
            font_dir.mkdir(exist_ok=True)

            # Generate text images
            print(f"\n📄 Generating text images...")
            self._generate_images_for_font(font_path, font_dir, font_name, images_per_font, is_digit=False)

            # Generate digit images
            print(f"\n🔢 Generating digit images...")
            self._generate_images_for_font(font_path, font_dir, font_name, digit_images_per_font, is_digit=True)

            print(f"\n✅ Completed: {font_name}")
            print(f"   - {images_per_font} text images")
            print(f"   - {digit_images_per_font} digit images")

        print("\n" + "=" * 60)
        print("🎉 Dataset generation completed successfully!")
        print(f"📂 Output path: {self.data_dir}")
        print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting Persian font dataset generation")
    print("=" * 60)

    generator = DatasetGenerator()
    generator.generate_dataset()
