import os
import PIL


IMG_DIR = 'data/images'


def rgba_to_rbg(img_data):
    """
    Takes a PIL image in RGBA format and returns it with a white background
    in RGB format.
    """
    rgba_pil = img_data.convert('RGBA')
    background = PIL.Image.new('RGBA', rgba_pil.size, (255, 255, 255))
    rgba_comp = PIL.Image.alpha_composite(background, rgba_pil)
    rgb_comp = rgba_comp.convert('RGB')
    return rgb_comp


def load_image(id, img_dir=IMG_DIR):
    """
    Loads an image from the punks dataset as an RGB PIL image.
    """
    img_path = f'{img_dir}/punk{"%04d" % id}.png'
    if not os.path.exists(img_path):
        raise Exception(f"ERROR: img_path failed to load {img_path}")
    pil_img = PIL.Image.open(img_path)
    return rgba_to_rbg(pil_img)


def write_image(img_data, filepath):
    """
    Simple method to save PIL image.
    """
    return img_data.save(filepath)


def load_mpimg(id, img_dir=IMG_DIR):
    """
    Loads an image using the legacy matplotlib imread function.
    
    Matplotlib wants people to use PIL instead, so this is here for
    compatibility with cpunks.
    """
    import matplotlib.image as mpimg
    return mpimg.imread(f'{img_dir}/punk{"%04d" % id}.png')


def write_mpimg(img_data, filepath):
    """
    Writes an image using the legacy matplotlib imsave function.
    
    Matplotlib wants people to use PIL instead, so this is here for
    compatibility with cpunks.
    """
    import matplotlib.image as mpimg
    return mpimg.imsave(filepath, img_data)
