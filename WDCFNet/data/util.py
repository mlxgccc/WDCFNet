
from PIL import Image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".bmp", ".JPG", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img
def load_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    return text
