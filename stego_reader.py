"""stego reader"""

import pathlib
import argparse
import contextlib

from PIL import Image

from stego import DataPixel


def picture_low_values(filename, lsbs):
    """main"""
    # filename = ""
    file = pathlib.Path(filename)
    img = Image.open(filename, "r")
    width, height = img.size
    shift = 8 - lsbs
    for ypos in range(height):
        print(f"row {ypos} of {height}", end="\r")
        for xpos in range(width):
            pos = (xpos, ypos)
            pix = img.getpixel(pos)
            red = (pix[0] << shift) % 2 ** 8
            green = (pix[1] << shift) % 2 ** 8
            blue = (pix[2] << shift) % 2 ** 8
            img.putpixel(pos, (red, green, blue, 255))
    with open(f"{file.stem}_lsb_{lsbs}.png", "wb") as file:
        img.save(file)


def make_xor(filename1, filename2):
    """make the xored image of two images."""
    img1 = Image.open(filename1)
    img2 = Image.open(filename2)
    width, height = img1.size
    for ypos in range(height):
        print(f"row {ypos} of {height}", end="\r")
        for xpos in range(width):
            pos = (xpos, ypos)
            pixel1 = img1.getpixel(pos)
            pixel2 = img2.getpixel(pos)
            xored = [pixel1[color] ^ pixel2[color] for color in range(3)]
            with contextlib.suppress(IndexError):
                xored.append(pixel1[3])
            img1.putpixel(pos, tuple(xored))
    print()
    file = pathlib.Path(filename1)
    with open(f"{file.stem}_xor.png", "wb") as file:
        img1.save(file)


def main(filename):
    """main"""
    # img = Image.open("Nut.png", "r")
    img = Image.open(filename, "r")
    width, height = img.size
    ints = []
    for ypos in range(height):
        print(f"row {ypos} of {height}", end="\r")
        for xpos in range(width):
            pos = (xpos, ypos)
            pixel = DataPixel(img.getpixel(pos))
            for val in pixel.low_value:
                ints.append(val)
    bits = "".join([f"{i:02b}" for i in ints])
    print(bits[:100]+"......"+bits[-100:])
    # bits = bits.rstrip("0")[:-2]
    bytes_ = [int(bits[i:i+8], 2) for i in range(0, len(bits), 8)]
    # with open("output.jpg", "wb") as file:
    #     file.write(bytes(bytes_))
    # print(bytes_)
    return bytes_


# def asymmetric():
#     with open("testkey") as file:
#         # print(help(RSA.import_key))
#         private_key = RSA.import_key(file.read(), passphrase="password")
#     cipher_rsa = PKCS1_OAEP.new(private_key)
#     key = cipher_rsa.decrypt(built[:256])
#     read_stream(key, built[256:])


def cli():
    """Command line interface reader"""
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Image to show LSBs")
    parser.add_argument("--lsbs", help="Amount of LSBs to be shiftet", type=int,
                        default=2)
    parser.add_argument("--compare", help="Image to xor with", default=None)
    arguments = parser.parse_args()
    if arguments.compare:
        make_xor(arguments.image, arguments.compare)
    else:
        picture_low_values(arguments.image, arguments.lsbs)


if __name__ == '__main__':
    # main("Nut.png")
    # picture_low_values()
    cli()
