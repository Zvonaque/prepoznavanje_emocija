# učitavanje biblioteka i argumenata
from PIL import Image
import sys
import os

py_filename = sys.argv[0]
folder_path = sys.argv[1]
new_folder = sys.argv[2]


# definirane funkcije za otvaranje slike
def open_image(path):
    newImage = Image.open(path)
    return newImage


# definirane funkcije za spremanje slike
def save_image(image, path):
    image.save(path, 'png')


# definirane funkcije za kreiranje slike dane dimenzije
def create_image(i, j):
    image = Image.new("RGB", (i, j), "white")
    return image


# definirane funkcije za dohvaćanje piksela iz slike
def get_pixel(image, i, j):
    # Inside image bounds?
    width, height = image.size
    if i > width or j > height:
      return None

    pixel = image.getpixel((i, j))
    return pixel


# definirane funkcije za kreiranje slike u nijansama sive
def convert_grayscale(image):
    width, height = image.size

    new = create_image(width, height)
    pixels = new.load()

    for i in range(width):
      for j in range(height):
        # Get Pixel
        pixel = get_pixel(image, i, j)

        # Get R, G, B values (This are int from 0 to 255)
        red =   pixel[0]
        green = pixel[1]
        blue =  pixel[2]

        gray = (red * 0.299) + (green * 0.587) + (blue * 0.114)

        pixels[i, j] = (int(gray), int(gray), int(gray))

    return new


# definirane funkcije za kreiranje slike s polutonovima
def convert_halftoning(image):
    width, height = image.size

    new = create_image(width, height)
    pixels = new.load()

    for i in range(0, width-1, 2):
      for j in range(0, height-1, 2):
        p1 = get_pixel(image, i, j)
        p2 = get_pixel(image, i, j + 1)
        p3 = get_pixel(image, i + 1, j)
        p4 = get_pixel(image, i + 1, j + 1)

        gray1 = (p1[0] * 0.299) + (p1[1] * 0.587) + (p1[2] * 0.114)
        gray2 = (p2[0] * 0.299) + (p2[1] * 0.587) + (p2[2] * 0.114)
        gray3 = (p3[0] * 0.299) + (p3[1] * 0.587) + (p3[2] * 0.114)
        gray4 = (p4[0] * 0.299) + (p4[1] * 0.587) + (p4[2] * 0.114)

        sat = (gray1 + gray2 + gray3 + gray4) / 4

        if sat > 223:
            pixels[i, j] = (255, 255, 255)  # White
            pixels[i, j + 1] = (255, 255, 255)  # White
            pixels[i + 1, j] = (255, 255, 255)  # White
            pixels[i + 1, j + 1] = (255, 255, 255)  # White
        elif sat > 159:
            pixels[i, j] = (255, 255, 255)  # White
            pixels[i, j + 1] = (0, 0, 0)  # Black
            pixels[i + 1, j] = (255, 255, 255)  # White
            pixels[i + 1, j + 1] = (255, 255, 255)  # White
        elif sat > 95:
            pixels[i, j] = (255, 255, 255)  # White
            pixels[i, j + 1] = (0, 0, 0)  # Black
            pixels[i + 1, j] = (0, 0, 0)  # Black
            pixels[i + 1, j + 1] = (255, 255, 255)  # White
        elif sat > 32:
            pixels[i, j] = (0, 0, 0)  # Black
            pixels[i, j + 1] = (255, 255, 255)  # White
            pixels[i + 1, j] = (0, 0, 0)  # Black
            pixels[i + 1, j + 1] = (0, 0, 0)  # Black
        else:
            pixels[i, j] = (0, 0, 0)  # Black
            pixels[i, j + 1] = (0, 0, 0)  # Black
            pixels[i + 1, j] = (0, 0, 0)  # Black
            pixels[i + 1, j + 1] = (0, 0, 0)  # Black

    return new


# definirane funkcije za vraćanje boje ovisno o kvadrantu i zasićenju
def get_saturation(value, quadrant):
    if value > 223:
        return 255
    elif value > 159:
        if quadrant != 1:
            return 255

        return 0
    elif value > 95:
        if quadrant == 0 or quadrant == 3:
            return 255

        return 0
    elif value > 32:
        if quadrant == 1:
            return 255

        return 0
    else:
        return 0


# definirane funkcije za kreiranje slike koja treperi
def convert_dithering(image):
    width, height = image.size

    new = create_image(width, height)
    pixels = new.load()

    for i in range(0, width-1, 2):
      for j in range(0, height-1, 2):
        p1 = get_pixel(image, i, j)
        p2 = get_pixel(image, i, j + 1)
        p3 = get_pixel(image, i + 1, j)
        p4 = get_pixel(image, i + 1, j + 1)

        red   = (p1[0] + p2[0] + p3[0] + p4[0]) / 4
        green = (p1[1] + p2[1] + p3[1] + p4[1]) / 4
        blue  = (p1[2] + p2[2] + p3[2] + p4[2]) / 4

        r = [0, 0, 0, 0]
        g = [0, 0, 0, 0]
        b = [0, 0, 0, 0]

        for x in range(0, 4):
          r[x] = get_saturation(red, x)
          g[x] = get_saturation(green, x)
          b[x] = get_saturation(blue, x)

        pixels[i, j]         = (r[0], g[0], b[0])
        pixels[i, j + 1]     = (r[1], g[1], b[1])
        pixels[i + 1, j]     = (r[2], g[2], b[2])
        pixels[i + 1, j + 1] = (r[3], g[3], b[3])

    return new


# definirane funkcije za kreiranje slike s primarnim bojama
def convert_primary(image):
    width, height = image.size

    new = create_image(width, height)
    pixels = new.load()

    for i in range(width):
      for j in range(height):
        pixel = get_pixel(image, i, j)

        red =   pixel[0]
        green = pixel[1]
        blue =  pixel[2]

        if red > 127:
            red = 255
        else:
            red = 0
        if green > 127:
            green = 255
        else:
            green = 0
        if blue > 127:
            blue = 255
        else:
            blue = 0

        pixels[i, j] = (int(red), int(green), int(blue))

    return new


# prolaz kroz sve slike u svim mapama zadane putanje
for subdir, dirs, files in os.walk(folder_path):
    for file in files:
        # Load Image (JPEG/JPG needs libjpeg to load)
        original = open_image(os.path.join(subdir, file))

        # pretvaranje slike u nijanse sive boje
        new = convert_grayscale(original)
        save_image(new, new_folder+file.split('.')[0]
        + '_grayscale'+'.png')

        # pretvaranje slike u polutonove
        new = convert_halftoning(original)
        save_image(new, new_folder+file.split('.')[0]
        + '_halftone'+'.png')

        # pretvaranje slike u treptaj
        new = convert_dithering(original)
        save_image(new, new_folder+file.split('.')[0]
        + '_dither'+'.png')

        # pretvaranje slike u primarne boje
        new = convert_primary(original)
        save_image(new, new_folder+file.split('.')[0]
        + '_primary'+'.png')
