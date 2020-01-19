"""Image stegomaker"""

import os  # pylint: disable=unused-import
import sys
import math
import random
# import cProfile
import hashlib
import getpass
import itertools
import contextlib
import collections
import pdb  # pylint: disable=unused-import
import time

import numpy
from PIL import Image
from Crypto.Cipher import AES
from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Util.Padding import pad
from Crypto.Util.Padding import unpad

# from stego_reader import main as reader_main

Rgb = collections.namedtuple("RGB", ["red", "green", "blue"])


class CleartextTooLarge(Exception):
    """Raise if plaintext is too large to fit into covertext"""


class DataPixel():
    """Data Pixel"""
    def __init__(self, pixelval):
        self.rgb = Rgb(pixelval[0], pixelval[1], pixelval[2])
        self.is_area = False
        self.is_next_to_area = False
        if len(pixelval) > 3:
            self.alpha = pixelval[3]
        else:
            self.alpha = None

    @property
    def value(self):
        """get pixel value"""
        rgb = self.rgb
        if self.alpha is not None:
            return (rgb.red, rgb.green, rgb.blue, self.alpha)
        return (rgb.red, rgb.green, rgb.blue)

    @property
    def low_value(self):
        """get value written in low-significance pixel"""
        rgb = self.rgb
        return (rgb.red % 4, rgb.green % 4, rgb.blue % 4)

    def __add__(self, other):
        red = (self.rgb.red >> 2 << 2)
        green = (self.rgb.green >> 2 << 2)
        blue = (self.rgb.blue >> 2 << 2)
        if isinstance(other, (BitString, BitString2)):
            red += next(other)
            green += next(other)
            blue += next(other)
            self.rgb = Rgb(red, green, blue)
        elif isinstance(other, tuple):
            red += other[0]
            green += other[1]
            blue += other[2]
            self.rgb = Rgb(red, green, blue)
        else:
            raise NotImplementedError
        return self

    def __eq__(self, other):
        return self.low_value == other.low_value

    def __repr__(self):
        return f"DataPixel({self.rgb})"

    def __str__(self):
        return self.__repr__()

    def set_color_lsb(self, color, value):
        """set the lsb on a color"""
        rgb = list(self.rgb)
        log2 = 2
        if color == "red":
            red = rgb[0] >> log2 << log2
            rgb[0] = red + value
        elif color == "green":
            green = rgb[1] >> log2 << log2
            rgb[1] = green + value
        elif color == "blue":
            blue = rgb[2] >> log2 << log2
            rgb[2] = blue + value
        self.rgb = Rgb(*rgb)


class BitString():
    """Bit String"""
    def __init__(self, bytestring):
        self.bytes = bytestring
        self.position = 0
        self.length = 2
        self.exhausted = False
        self.iterator = False

    def __iter__(self):
        self.position = 0
        self.exhausted = False
        self.iterator = True
        return self

    def __next__(self):
        # pdb.set_trace()
        if self.exhausted and not self.iterator:
            return 0
        # print(self.bytes[self.position // 4])
        # byte = int.from_bytes(self.bytes[self.position // 4], "little")
        # import ipdb; ipdb.set_trace()
        try:
            byte = self.bytes[self.position // 4]
        except IndexError:
            self.exhausted = True
            if self.iterator:  # pylint: disable=no-else-raise
                raise StopIteration
            else:
                return 10
        offset = (3 - self.position % 4) * 2
        group = (byte >> offset) % 4
        self.position += 1
        return group


class BitString2():
    """nonfilling bitstring"""
    def __init__(self, bytestring):
        self.bytes = bytestring

    def required_pixels(self, lsbs=2):
        """Value how many pixels are required to save the bitstring."""
        return math.ceil(len(self.bytes) * 8 / 3 / lsbs)

    def __iter__(self):
        for byte in self.bytes:
            # num = int.from_bytes(byte, "big")
            # String lösung
            for chunk in chunks(f"{byte:08b}", 2):
                yield int(chunk, base=2)
            # Mathe Lösung
            # yield byte >> 6
            # yield (byte >> 4) & 3
            # yield (byte >> 2) & 3
            # yield byte & 3
    #
    # def __next__(self):
    #     binary = int.from_bytes(self.bytes, "little")
    #     for chunk in chunks(f"{binary:08b}", 2):
    #         yield int(chunk, base=2)


def chunks(lst, num):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), num):
        yield lst[i:i + num]


# print("Start")

# for i in BitString(b"\xFF\x05"):
#     print(f"{i:02b}", end="")
# print()


def encrypt(data, key):
    """encrypt"""
    cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce
    result = cipher.encrypt_and_digest(data)[0]
    return result, nonce


def decrypt(ciphertext, nonce, key):
    """decrypt aes"""
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    plaintext = cipher.decrypt(ciphertext)
    return plaintext


def keygen(key):
    """make a key"""
    # key = os.urandom(256 // 8)
    hasher = hashlib.sha256()
    hasher.update(key.encode())
    key = hasher.digest()
    print(f"Generated {key=}")
    return key


def write_to_image(data, img, target_name):
    """Write some to an image"""
    width, height = img.size
    databit = BitString(data)
    # matrix = numpy.empty(img.size, DataPixel)
    # for ypos in range(height):
    #     print(f"loading row {ypos} of {height}", end="\r")
    #     for xpos in range(width):
    #         pos = (xpos, ypos)
    #         matrix[pos] = DataPixel(img.getpixel(pos))
    # print(matrix)
    # find_editable_pixels(matrix)
    for ypos in range(height):
        print(f"row {ypos} of {height}", end="\r")
        for xpos in range(width):
            pos = (xpos, ypos)
            pixel = DataPixel(img.getpixel(pos))
            pixel += databit
            img.putpixel(pos, pixel.value)
    print()
    with open(target_name, "wb") as file:
        img.save(file)


def random_bits(bitnum):
    """Get random bits"""
    return os.urandom(1)[0] % 2 ** bitnum


def write_header(header, array):
    """Write header into image"""
    needed_header_pixels = math.ceil(len(header) * 8 / 2 / 3)
    # print(needed_header_pixels)
    bits = iter(BitString2(header))
    # print(f"{needed_header_pixels=}   ---- write_header")
    for idx in itertools.islice(numpy.ndindex(array.shape[:2]),
                                needed_header_pixels):
        pixel = DataPixel(array[idx])
        with contextlib.suppress(StopIteration):
            pixel.set_color_lsb("red", next(bits))
            pixel.set_color_lsb("green", next(bits))
            pixel.set_color_lsb("blue", next(bits))
        # print(f"{red:02b}{green:02b}{blue:02b}", end="")
        array[idx] = pixel.value
        # for val in pixel.value[:3]:
        #     print(f"{val%4:02b}", end="")
        # print()


def write_to_image2(header, data, img, randomseed, target_name):
    """Write some to an image"""
    array = numpy.array(img)
    write_header(header, array)
    needed_header_pixels = math.ceil(len(header) * 8 / 2 / 3)
    needed_message_pixels = math.ceil(len(data) * 8 / 2 / 3)
    # print(f"{len(data)=}")
    # print(f"{data[:100]=}")

    random.seed(randomseed)
    # print(f"{needed_header_pixels=}   ---- write_to_image2")
    indexes = list(numpy.ndindex(array.shape[:2]))[needed_header_pixels:]
    target_pixels = random.sample(indexes, needed_message_pixels)
    # print(f"{needed_message_pixels=}")
    # print(f"{hash(tuple(target_pixels))=}")
    # print(f"{target_pixels[:10]=}")
    # print(f"{target_pixels[-10:]=}")
    bits = iter(BitString2(data))
    for position in target_pixels:
        pixel = DataPixel(array[position])
        with contextlib.suppress(StopIteration):
            # print(f"{pixel=}")
            pixel.set_color_lsb("red", next(bits))
            pixel.set_color_lsb("green", next(bits))
            pixel.set_color_lsb("blue", next(bits))
            # print(f"{pixel=}")
        array[position] = pixel.value
    img = Image.fromarray(array)
    with open(target_name, "wb") as file:
        img.save(file)


def grouper(iterable, members, fillvalue=None):
    """Cut iterator into chunks."""
    args = [iter(iterable)] * members
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def build_stream(data, key, target_size):
    """build a stream"""
    ciphertext, nonce = encrypt(data, key)
    length = len(ciphertext)
    length = str(length).encode()
    length = pad(length, 32)
    lencipher, lennonce = encrypt(length, key)
    randompad = os.urandom(target_size - len(nonce) - len(lennonce) -
                           len(lencipher) - len(ciphertext))
    ciphertext = ciphertext + randompad
    # print(nonce, lennonce, lencipher, ciphertext, randompad)
    # return nonce, lennonce, lencipher, ciphertext
    return nonce + lennonce + lencipher + ciphertext


def build_with_randomseed(data, key, seed, asym_cipher=None):
    """build a stream including a randomseed. returns header and ciphertext."""
    ciphertext, nonce = encrypt(data, key)
    length = len(ciphertext)
    length = str(length).encode()
    length = pad(length, 32)
    lencipher, lennonce = encrypt(length, key)
    randcipher, randnonce = encrypt(seed, key)
    header = lennonce + lencipher + randnonce + randcipher + nonce
    if asym_cipher is not None:
        header = asym_cipher + header
    return header, ciphertext


# def read_stream(key, nonce, lennonce, lencipher, ciphertext):
def read_stream(key, cipher):
    """read the message"""
    nonce = cipher[:16]
    lennonce = cipher[16:32]
    lencipher = cipher[32:64]
    ciphertext = cipher[64:]
    length = decrypt(lencipher, lennonce, key)
    length = int(unpad(length, 32).decode())
    ciphertext = ciphertext[:length]
    data = decrypt(ciphertext, nonce, key)
    # print(data)
    return data


def read_stream2(key, cipher, array):
    """Finds the header in the ciphertext and gives to array."""
    lennonce = cipher[:16]
    lencipher = cipher[16:48]
    randnonce = cipher[48:64]
    randcipher = cipher[64:96]
    nonce = cipher[96:112]
    length = decrypt(lencipher, lennonce, key)
    length = int(unpad(length, 32).decode())
    randomseed = decrypt(randcipher, randnonce, key)
    needed_header_pixels = math.ceil((112 + 256) * 8 / 2 / 3)  # 256: key, 112: field lengths
    needed_message_pixels = math.ceil(length * 8 / 2 / 3)
    indexes = list(numpy.ndindex(array.shape[:2]))[needed_header_pixels:]
    random.seed(randomseed)
    indexes = random.sample(indexes, needed_message_pixels)
    ints = []
    for index in indexes:
        for val in DataPixel(array[index]).low_value:
            ints.append(val)
    bits = "".join([f"{i:02b}" for i in ints])
    bytes_ = [int(bits[i:i+8], 2) for i in range(0, len(bits), 8)]
    encrypted = bytes(bytes_)
    data = decrypt(encrypted, nonce, key)
    return data


def encrypt_symmetric(source, target, payload, password):
    """main"""
    img = Image.open(source)
    key = keygen(password)
    # encrypt(b"Geheime Nachricht!", "passwort")
    # sys.exit()
    # print(dir(img))
    # print(img.size)
    width, height = img.size
    target_size = (width * height) // 8 * 2 * 3

    with open("jste.py", "rb") as file:
        data = file.read()
    if len(data) > target_size:
        print("File too large!")
        raise CleartextTooLarge
    built = build_stream(data, key, target_size)
    read_stream(key, built)

    # databit = BitString(data)
    # ints = []
    # for _ in range(target_size):
    #     ints.append((next(databit), next(databit), next(databit)))

    # print(f"{data[0]:b}")
    # target_name = "Nut_output.png"
    target_name = target
    write_to_image(built, img, target_name)


def encrypt_asymmetric(image_name, public_key_filename, data_file):
    """main"""
    img = Image.open(image_name)
    with open(public_key_filename, "r") as file:
        extern_key = file.read()
    rsakey = RSA.import_key(extern_key, passphrase=None)
    key = os.urandom(32)
    cipher_rsa = PKCS1_OAEP.new(rsakey)
    encrypted_key = cipher_rsa.encrypt(key)
    width, height = img.size
    target_size = (width * height) // 8 * 2 * 3

    with open(data_file, "rb") as file:
        data = file.read()
    if len(data) > target_size:
        print("File too large!")
        sys.exit(1)
    built = encrypted_key + build_stream(data, key, target_size - 256)

    # Decrypt
    with open("testkey") as file:
        # print(help(RSA.import_key))
        private_key = RSA.import_key(file.read(), passphrase="password")
    cipher_rsa = PKCS1_OAEP.new(private_key)
    key = cipher_rsa.decrypt(built[:256])
    read_stream(key, built[256:])

    target_name = "castle_rsa_random.png"
    # write_to_image(built, img, target_name)
    randomseed = os.urandom(32)
    # print(data)
    header, ciphertext = build_with_randomseed(data, key, randomseed,
                                               asym_cipher=encrypted_key)
    # print(f"{ciphertext[:100]=}")
    write_to_image2(header, ciphertext, img, randomseed, target_name)
    print(decrypt_asymmetric(target_name, "testkey", "password"))
    # decrypt_asymmetric(target_name, "testkey", "password")


def read_bytes_in_image(filename):
    """main"""
    # img = Image.open("Nut.png", "r")
    img = Image.open(filename, "r")
    width, height = img.size
    ints = []
    # matrix = numpy.empty(img.size, DataPixel)
    # for ypos in range(height):
    #     print(f"loading row {ypos} of {height}", end="\r")
    #     for xpos in range(width):
    #         pos = (xpos, ypos)
    #         matrix[pos] = DataPixel(img.getpixel(pos))
    # print(matrix)
    counter = 0
    for ypos in range(height):
        print(f"processing row {ypos} of {height}", end="\n")
        if counter > 2:
            break
        counter += 1
        for xpos in range(width):
            pos = (xpos, ypos)
            pixel = DataPixel(img.getpixel(pos))
            for val in pixel.low_value:
                ints.append(val)
    print()
    bits = "".join([f"{i:02b}" for i in ints])
    # print(f"{bits[:100]}...   ...{bits[-100:]}")
    # bits = bits.rstrip("0")[:-2]
    bytes_ = [int(bits[i:i+8], 2) for i in range(0, len(bits), 8)]
    # with open("output.jpg", "wb") as file:
    #     file.write(bytes(bytes_))
    # print(bytes_)
    return bytes(bytes_)


def decrypt_symmetric(filename, password):
    """read symmetrically encrypted data in stego image"""
    built = read_bytes_in_image(filename)
    key = keygen(password)
    return read_stream(key, built)


def decrypt_asymmetric(filename, keyfile, password=None, *, interact=False):
    """read asymmetrically encrypted data in stego image."""
    with open(keyfile) as file:
        if interact:
            password = getpass.getpass()
        private_key = RSA.import_key(file.read(), passphrase=password)
    built = read_bytes_in_image(filename)
    array = numpy.array(Image.open(filename))
    cipher_rsa = PKCS1_OAEP.new(private_key)
    key = cipher_rsa.decrypt(built[:256])  # runtime: 0.015s on Desktop
    return read_stream2(key, built[256:], array)


def set_lsb(pixel, value):
    """set the LSBs"""
    pixel[0] = pixel[0] >> 2 << 2
    pixel[1] = pixel[1] >> 2 << 2
    pixel[2] = pixel[2] >> 2 << 2
    pixel[0] += value
    pixel[1] += value
    pixel[2] += value


def find_editable_pixels(matrix):
    """mark elements as writable or not"""
    queue = collections.deque(maxlen=matrix.shape[0])
    for element in numpy.nditer(matrix):
        queue.append(element.low_value)
    # others = ((ypos-1, xpos-1), (ypos, xpos-1), (ypos-1, xpos), (ypos-1, xpos+2))


def mark(filename):
    """mark an image"""
    array = numpy.array(Image.open(filename))
    height = array.shape[0]
    for idx_i, i in enumerate(array):
        print(f"progress: {idx_i / height * 100:3.0f}%", end="\r")
        for idx_j, j in enumerate(i):
            set_lsb(j, (idx_i // 10 % 2 ^ idx_j // 10 % 2) * 3)
    img = Image.fromarray(array)
    print()
    with open("arrayed_lsb2.png", "wb") as file:
        img.save(file)


# def asymetric_write():
#     """asymmetric key"""
#     with open("testkey.pub", "r") as file:
#         extern_key = file.read()
#     key = RSA.import_key(extern_key, passphrase=None)


if __name__ == '__main__':
    # stego_encrypt("symmetric")
    # stego_encrypt_asymmetric()
    # decrypt_asymmetric("Nut_rsa.png", "testkey", "password")
    # mark("Nut.png")
    start = time.time()
    encrypt_asymmetric("castle.bmp", "testkey.pub", "jste.py")
    print(f"Finished in {time.time() - start:2.5}s")
    # import cProfile
    # cProfile.run("encrypt_asymmetric()")
