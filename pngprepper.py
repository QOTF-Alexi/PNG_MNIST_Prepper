import struct
import zlib
import pathlib
import numpy as np

""" Read the PNG. """
def read_png(path):
    with open(path, "rb") as f:
        data = f.read()

    # Validate that a valid PNG is given.
    # Reference: https://www.w3.org/TR/png/#5PNG-file-signature
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos = 8

    width = height = None
    bit_depth = None
    color_type = None
    idat = b""

    # Parse the chunks of the PNG. A chunk is made up of 12 bytes.
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos+4])[0]
        ctype = data[pos+4:pos+8]
        chunk = data[pos+8:pos+8+length]
        pos += 12 + length

        # A PNG is made up of, among others, IHDR (Image Header), IDAT (Image Data) and IEND (Image Trailer).
        # Only these parts are of interest to us.
        # Reference: https://www.w3.org/TR/PNG-Chunks.html#C.Critical-chunks
        if ctype == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk[:10])
        elif ctype == b"IDAT":
            idat += chunk
        elif ctype == b"IEND":
            break

    return width, height, bit_depth, color_type, idat

"""
Remove the scanline filter from the PNG
Reference: https://www.w3.org/TR/PNG-Filters.html
"""
def unfilter(raw, width, height, bpp):
    stride = width * bpp
    out = np.zeros((height, stride), dtype=np.uint8)

    i = 0
    for y in range(height):
        f = raw[i]
        i += 1
        scan = np.frombuffer(raw[i:i+stride], dtype=np.uint8)
        i += stride

        # No filter.
        if f == 0:
            out[y] = scan

        # Sub filter.
        elif f == 1:
            row = scan.astype(np.int16)
            row[bpp:] = (row[bpp:] + row[:-bpp]) & 0xFF
            out[y] = row.astype(np.uint8)

        # Up filter.
        elif f == 2:
            if y == 0:
                out[y] = scan
            else:
                out[y] = (scan.astype(np.int16) + out[y-1].astype(np.int16)) & 0xFF

        # Average filter.
        elif f == 3:
            row = scan.astype(np.int16)
            if y == 0:
                left = np.zeros(stride, dtype=np.int16)
                up = np.zeros(stride, dtype=np.int16)
            else:
                left = np.zeros(stride, dtype=np.int16)
                left[bpp:] = out[y].astype(np.int16)[:-bpp]
                up = out[y-1].astype(np.int16)

            avg = ((left + up) >> 1) & 0xFF
            out[y] = (row + avg) & 0xFF

        # Paeth filter.
        elif f == 4:
            row = scan.astype(np.int16)

            if y == 0:
                a = np.zeros(stride, dtype=np.int16)
                b = np.zeros(stride, dtype=np.int16)
                c = np.zeros(stride, dtype=np.int16)
            else:
                a = np.zeros(stride, dtype=np.int16)
                a[bpp:] = out[y].astype(np.int16)[:-bpp]
                b = out[y-1].astype(np.int16)
                c = np.zeros(stride, dtype=np.int16)
                c[bpp:] = out[y-1].astype(np.int16)[:-bpp]

            p = a + b - c
            pa = np.abs(p - a)
            pb = np.abs(p - b)
            pc = np.abs(p - c)

            pr = np.where((pa <= pb) & (pa <= pc), a,
                 np.where(pb <= pc, b, c))

            out[y] = (row + pr) & 0xFF

    return out

""" Convert to grayscale. """
def to_grayscale(pixels, width, height, colour_type):
    # Voor zwart-wit
    if colour_type == 0:
        return pixels.reshape(height, width)

    # Let's assume ITU-R Recommendation BT.601 is used.
    # Reference: https://www.itu.int/rec/R-REC-BT.601/

    ## Colour types reference: https://www.w3.org/TR/PNG-Chunks.html#C.Critical-chunks

    # For true colour
    if colour_type == 2:
        r = pixels[:, 0::3]
        g = pixels[:, 1::3]
        b = pixels[:, 2::3]
        return (0.299*r + 0.587*g + 0.114*b).astype(np.uint8)

    # For true colour + alpha
    if colour_type == 6:
        r = pixels[:, 0::4]
        g = pixels[:, 1::4]
        b = pixels[:, 2::4]
        return (0.299*r + 0.587*g + 0.114*b).astype(np.uint8)

    raise ValueError("Unsupported color type")

""" Resize to 28x28 or nearest neighbour. """
def resize_28x28(img):
    h, w = img.shape
    ys = (np.linspace(0, h-1, 28)).astype(int)
    xs = (np.linspace(0, w-1, 28)).astype(int)
    return img[ys][:, xs]

""" Invert the image if required, to get a white number on a black background. """
def auto_invert(img):
    corners = np.array([
        img[0,0], img[0,-1],
        img[-1,0], img[-1,-1]
    ])
    if corners.mean() > 0.5:
        return 1.0 - img
    return img

""" For an NN you need a flattened vector. """
def flatten_28x28(img):
    return img.reshape(1, 784)

def load_png_as_28x28_uint8(path: str) -> np.uint8:
    w, h, bd, ct, idat = read_png(path)
    raw = zlib.decompress(idat)

    bpp = {0:1, 2:3, 6:4}[ct]
    pixels = unfilter(raw, w, h, bpp)
    gray = to_grayscale(pixels, w, h, ct)
    small = resize_28x28(gray)
    small_normalised = small.astype(np.float32) / 255.0
    inverted = auto_invert(small_normalised)
    flattened = flatten_28x28(inverted)

    return flattened.astype(np.uint8)

# Usage example:
infile = f"{pathlib.Path().resolve()}/input.png"
preprocessed_img = load_png_as_28x28_uint8(infile)
