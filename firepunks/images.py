import os
import PIL


IMG_DIR = 'data/images'


def rgba_to_rbg(img_data):
    rgba_pil = img_data.convert('RGBA')
    background = PIL.Image.new('RGBA', rgba_pil.size, (255, 255, 255))
    rgba_comp = PIL.Image.alpha_composite(background, rgba_pil)
    rgb_comp = rgba_comp.convert('RGB')
    return rgb_comp


def load_image(id, img_dir=IMG_DIR):
    img_path = f'{img_dir}/punk{"%04d" % id}.png'
    if not os.path.exists(img_path):
        raise Exception(f"ERROR: img_path failed to load {img_path}")
    pil_img = PIL.Image.open(img_path)
    return rgba_to_rbg(pil_img)


def write_image(img_data, filepath):
    return img_data.save(filepath)


def load_mpimg(id, img_dir=IMG_DIR):
    import matplotlib.image as mpimg
    return mpimg.imread(f'{img_dir}/punk{"%04d" % id}.png')


def write_mpimg(img_data, filepath):
    import matplotlib.image as mpimg
    return mpimg.imsave(filepath, img_data)
