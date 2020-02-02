"""Stego basic functions."""

import os  # pylint: disable=unused-import
import sys
import math
import random
# import cProfile
import hashlib
import itertools
import contextlib
import collections
import pdb  # pylint: disable=unused-import

import numpy
from PIL import Image
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Util.Padding import unpad

# from stego_reader import main as reader_main

Rgb = collections.namedtuple("RGB", ["red", "green", "blue"])
RSA_ENCRYPTED_LENGTH = 256  # Bytes
NONCE_LENGTH = 16  # Bytes
CLUSTER_ITERATION_LIMIT = 1000


class CleartextTooLarge(Exception):
    """Raise if plaintext is too large to fit into covertext"""


class DataPixel():
    """Data Pixel"""
    def __init__(self, pixelval):
        self.rgb = Rgb(pixelval[0], pixelval[1], pixelval[2])
        if pixelval[-1] > 255:  # Metadata saved
            self.state = pixelval[-1]
        else:
            self.state = None
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


class BitString2():  # pylint: disable=too-few-public-methods
    """nonfilling bitstring"""
    def __init__(self, bytestring):
        self.bytes = bytestring

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


def needed_pixels(data, lsbs=2):
    """Value how many pixels are required to save the bitstring."""
    if isinstance(data, int):
        return math.ceil(data * 8 / 3 / lsbs)
    return math.ceil(len(data) * 8 / 3 / lsbs)

# for i in BitString(b"\xFF\x05"):
#     print(f"{i:02b}", end="")
# print()


def int_to_bytes(int_, padlength=32):
    """Convert int to string, then to bytes, then pad it."""
    as_str = str(int_)
    as_bytes = as_str.encode()
    padded = pad(as_bytes, padlength)
    return padded


def bytes_to_int(bytes_, padlength=32):
    """Convert int to string, then to bytes, then pad it."""
    # print(f"{bytes_=}")
    unpadded = unpad(bytes_, padlength)
    as_str = unpadded.decode()
    int_ = int(as_str)
    return int_


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


def random_sample(iterable, *_):
    """exhaustive version of random.sample without max amount"""
    copy = iterable.copy()
    number = len(copy) - 1
    new = []
    while number >= 0:
        selection = random.randint(0, number)
        new.append(copy.pop(selection))
        number -= 1
    return new


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


def write_header_avoid_clusters(header, array, positions):
    """Write header into image, but only in given locations."""
    needed_header_pixels = needed_pixels(len(header))
    # print(needed_header_pixels)
    bits = iter(BitString2(header))
    # print(f"{needed_header_pixels=}   ---- write_header")
    for idx in itertools.islice(positions, needed_header_pixels):
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
    needed_header_pixels = needed_pixels(header)
    needed_message_pixels = needed_pixels(data)
    print(f"{randomseed=}")

    random.seed(randomseed)
    # print(f"{needed_header_pixels=}   ---- write_to_image2")
    indexes = list(numpy.ndindex(array.shape[:2]))[needed_header_pixels:]
    target_pixels = random.sample(indexes, needed_message_pixels)
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


def write_to_image_avoid_clusters(header, data, img, randomseed, target_name):
    """Write some to an image"""
    array = numpy.array(img)
    needed_header_pixels = needed_pixels(header)
    needed_message_pixels = needed_pixels(data)
    print(f"{len(data)=}")

    random.seed(randomseed)
    # print(f"{randomseed=}")
    # print(f"{needed_header_pixels=}   ---- write_to_image2")
    usables = find_editable_pixels(array)
    # indexes = list(numpy.ndindex(array.shape[:2]))
    # indexes = set(indexes)
    # indexes -= set(unusables)
    # indexes = list(sorted(indexes))
    indexes = usables
    print(f"Hash of pixels found: {hash(tuple(indexes))}")
    # print(f"{indexes[:100]=}")
    write_header_avoid_clusters(header, array, indexes[:needed_header_pixels])
    target_pixels = random_sample(indexes[needed_header_pixels:],
                                  needed_message_pixels)
    print(f"{hash(tuple(target_pixels[:500]))=}")
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


def build_with_randomseed_short_header(data, key, seed, cipher_rsa=None):
    """build a stream including a randomseed. returns header and ciphertext.

    This makes the header as short as possible (only randomseed).
    Asymmetric format:
    - Header format:
    - - RSA Randomseed
    - Ciphertext format:
    - - RSA Len, RSA aes-key, PLAIN nonce for AES, AES ciphertext
    """
    ciphertext, nonce = encrypt(data, key)
    length = int_to_bytes(len(ciphertext))
    print(f"{length=}")
    if cipher_rsa:
        randcipher = cipher_rsa.encrypt(seed)
        lencipher = cipher_rsa.encrypt(length)
        symcipher = cipher_rsa.encrypt(key)
    else:
        raise NotImplementedError
        # randcipher, randnonce = encrypt(seed, key)
        # randcipher = randcipher + randnonce
    header = randcipher
    print(f"{len(ciphertext)=}")
    ciphertext = lencipher + symcipher + nonce + ciphertext
    print(f"{len(ciphertext)=}")
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


def read_stream2(key, cipher, array):  # pylint: disable=too-many-locals
    """Finds the header in the ciphertext and gives to array."""
    lennonce = cipher[:16]
    lencipher = cipher[16:48]
    randnonce = cipher[48:64]
    randcipher = cipher[64:96]
    nonce = cipher[96:112]
    # lennonce + lencipher + randnonce + randcipher + nonce
    length = decrypt(lencipher, lennonce, key)
    length = int(unpad(length, 32).decode())
    randomseed = decrypt(randcipher, randnonce, key)
    needed_header_pixels = needed_pixels(112 + 256)  # 256: key, 112: field lengths
    needed_message_pixels = needed_pixels(length)
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


def read_stream3(cipher_rsa, array, avoid_clusters=False):
    # pylint: disable=too-many-locals
    """Finds the header in the ciphertext and gives to array."""
    all_pixels = list(numpy.ndindex(array.shape[:2]))
    if avoid_clusters:
        usables = find_editable_pixels(array)
        # print(f"{len(unusables)=}")
        # indexes = list(numpy.ndindex(array.shape[:2]))
        # indexes = set(indexes)
        # indexes -= set(unusables)
        # indexes = list(sorted(indexes))
        # all_pixels = indexes
        all_pixels = usables
    print(f"Hash of pixels found: {hash(tuple(all_pixels))}")
    # print(f"{all_pixels[:100]=}")
    # key = cipher_rsa.decrypt(built[:256])
    data = read_bytes_in_image2(array, all_pixels)
    ciphered_key, cipher = data[:256], data[256:]
    print(f"{hash(ciphered_key)=}")
    print(f"{len(ciphered_key)=}")
    key = cipher_rsa.decrypt(ciphered_key)
    lennonce = cipher[:16]
    lencipher = cipher[16:48]
    randnonce = cipher[48:64]
    randcipher = cipher[64:96]
    nonce = cipher[96:112]
    # lennonce + lencipher + randnonce + randcipher + nonce
    length = decrypt(lencipher, lennonce, key)
    length = int(unpad(length, 32).decode())
    randomseed = decrypt(randcipher, randnonce, key)
    needed_header_pixels = needed_pixels(112 + 256)  # 256: key, 112: field lengths
    needed_message_pixels = needed_pixels(length)
    indexes = list(all_pixels)[needed_header_pixels:]
    random.seed(randomseed)
    print(f"{randomseed=}")
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


def read_stream4(cipher_rsa, array, avoid_clusters=False):
    # pylint: disable=too-many-locals
    """Finds the header in the ciphertext and gives to array."""
    all_pixels = list(numpy.ndindex(array.shape[:2]))
    if avoid_clusters:
        usables = find_editable_pixels(array)
        all_pixels = usables
    print(f"Hash of pixels found: {hash(tuple(all_pixels))}")
    # print(f"{all_pixels[:100]=}")
    # key = cipher_rsa.decrypt(built[:256])

    # Find the fixed header
    data = read_bytes_in_image2(array, all_pixels)
    ciphered_random = data[:RSA_ENCRYPTED_LENGTH]
    randomseed = cipher_rsa.decrypt(ciphered_random)
    # print(f"{hash(ciphered_key)=}")
    # print(f"{len(ciphered_key)=}")
    needed_header_pixels = needed_pixels(RSA_ENCRYPTED_LENGTH)
    # print(f"{needed_header_pixels=}")
    indexes = list(all_pixels)[needed_header_pixels:]
    random.seed(randomseed)
    # print(f"{randomseed=}")
    indexes = random_sample(indexes, 2680)
    ints = []
    for index in indexes:
        for val in DataPixel(array[index]).low_value:
            ints.append(val)
    bits = "".join([f"{i:02b}" for i in ints])
    bytes_ = [int(bits[i:i+8], 2) for i in range(0, len(bits), 8)]
    encrypted = bytes(bytes_)
    # RSA Len, RSA aes-key, PLAIN nonce for AES, AES ciphertext
    length = encrypted[:RSA_ENCRYPTED_LENGTH]
    aes_key = encrypted[RSA_ENCRYPTED_LENGTH:RSA_ENCRYPTED_LENGTH * 2]
    nonce = encrypted[RSA_ENCRYPTED_LENGTH * 2:
                      RSA_ENCRYPTED_LENGTH * 2 + NONCE_LENGTH]
    length = cipher_rsa.decrypt(length)
    length = bytes_to_int(length)
    cipher = encrypted[RSA_ENCRYPTED_LENGTH * 2 + NONCE_LENGTH:
                       RSA_ENCRYPTED_LENGTH * 2 + NONCE_LENGTH + length]
    aes_key = cipher_rsa.decrypt(aes_key)
    data = decrypt(cipher, nonce, aes_key)
    return data


def read_bytes_in_image(filename):
    """main"""
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


def read_bytes_in_image2(array, indexes):
    """main"""
    ints = []
    for counter, position in enumerate(indexes):
        print(f"processing pixel {counter}", end="\r")
        # if counter >= 1000:
        #     break
        pixel = DataPixel(array[position])
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


def set_lsb(pixel, value):
    """set the LSBs"""
    pixel[0] = (pixel[0] >> 2 << 2) + value
    pixel[1] = (pixel[1] >> 2 << 2) + value
    pixel[2] = (pixel[2] >> 2 << 2) + value


def get_lsb(pixel):
    "More efficient lsb getting"
    red = pixel[0] & 3
    green = pixel[1] & 3
    blue = pixel[2] & 3
    return red, green, blue


def get_lsb_single(pixel):
    "More efficient lsb getting"
    red = pixel[0] & 3
    green = pixel[1] & 3
    blue = pixel[2] & 3
    return ((red * 4) + green) * 4 + blue


def binary_lsb(reduced_lsb, new_color):
    "More efficient lsb getting"
    second_lsb = new_color & 3
    return (reduced_lsb * 4) + second_lsb


def surrounding_pixels(array, coords, mode="direct neighborhood"):
    """Get all the surrounding pixels in an array."""
    shape = array.shape
    new_positions = []
    first, second = coords[0], coords[1]
    if mode == "full neighborhood":
        for ypos in range(first - 1, first + 2):
            for xpos in range(second - 1, second + 2):
                # print(xpos, ypos)
                # if xpos == first and ypos == second:
                #     continue
                if ypos >= shape[0] or xpos >= shape[1] or xpos < 0 or ypos < 0:
                    continue
                new_positions.append((ypos, xpos))
    elif mode == "direct neighborhood":
        positions = ((first - 1, second), (first + 1, second), (first, second),
                     (first, second - 1), (first, second + 1))
        for xpos, ypos in positions:
            if xpos >= shape[0] or ypos >= shape[1] or xpos < 0 or ypos < 0:
                continue
            new_positions.append((xpos, ypos))
    return new_positions


def mark_cluster_unusable(unusable: numpy.array, landlocked_unusable: numpy.array,
                          position: tuple, array: numpy.array):
    """mark a cluster of a color as unusable"""
    initial_lsb = array[position]

    candidates = {position}
    # candidates = collections.deque([position])
    # i = 0
    visited = numpy.zeros(unusable.shape)
    # iteration = 0
    while candidates:
        # if iteration >= CLUSTER_ITERATION_LIMIT:
        #     break
        # iteration += 1
        # print(f"depth: {i}", end="\r")
        # print(candidates)
        # i += 1
        current = candidates.copy()
        candidates = set()
        # current -= visited
        # current[visited] = False
        for candidate in current:
            if visited[candidate]:
                continue
            # if candidate == (100, 400):
            #     import ipdb; ipdb.set_trace()
            # if candidate in unusable:
            #     continue
            if array[candidate] == initial_lsb:
                unusable[candidate] = True
                visited[candidate] = True
                surroundings = surrounding_pixels(array, candidate)
                amount_pixels_same_color = 0  # including self
                for surrounding_pixel in surroundings:
                    # console_image.draw(array, unusable, surrounding_pixel)
                    if unusable[surrounding_pixel]:
                        amount_pixels_same_color += 1
                        # technically not same color, but landlocked anyway
                        continue
                    if visited[surrounding_pixel]:
                        print("PANIC PANIC "*10)
                    # time.sleep(0.5)
                    if array[surrounding_pixel] == initial_lsb:
                        amount_pixels_same_color += 1
                        candidates.add(surrounding_pixel)
                        unusable[surrounding_pixel] = True
                # if candidate == (400, 100):
                #     import ipdb; ipdb.set_trace()
                if amount_pixels_same_color == len(surroundings):
                    # Surrounded totally by same-colored pixels
                    landlocked_unusable[candidate] = True
        # candidates = set(surroundings)
    # print()


def find_editable_pixels(array):  # pylint:disable=too-many-locals
    """mark elements as writable or not"""
    # low_array = numpy.empty((array.shape[:2]))
    # for position in numpy.ndindex(array.shape[:2]):
    #     # This takes a very long time, why?
    #     low_array[position] = get_lsb_single(array[position])

    ufunc = numpy.frompyfunc(binary_lsb, 2, 1)
    low_array = ufunc.reduce(array, -1, initial=0)
    unusable = numpy.zeros(array.shape[:2], dtype=bool)
    landlocked_unusable = numpy.zeros(array.shape[:2], dtype=bool)
    # queue = collections.deque(maxlen=array.shape[0])
    indexes = numpy.ndindex(array.shape[:2])
    indexes = list(indexes)
    pixels = array.shape[0] * array.shape[1]
    interstep = pixels // 1000
    for idx, position in enumerate(indexes):
        if idx % interstep == 0:
            print(f"{(idx + 1) / (pixels) * 100:6.2f}%", end="\r")
        if unusable[position]:
            continue
        surroundings = surrounding_pixels(array, position)
        neighbors = [low_array[coord] for coord in surroundings]
        # center is included in surroundings
        if len(set(neighbors)) == 1:
            mark_cluster_unusable(unusable, landlocked_unusable, position, low_array)
    print("\nFinished searching editable pixels.")
    border_pixels = numpy.argwhere(unusable & ~landlocked_unusable)
    # Make pixels next to clusters unusable too
    for position in border_pixels:
        for surround_position in surrounding_pixels(array, position):
            unusable[surround_position] = True

    # Show landlocked pixels
    # temp = array.copy()
    # # temp[landlocked_unusable] = [255, 0, 255, 255]
    # temp[landlocked_unusable] = [255, 0, 255]  # bmp
    # # for landlocked in landlocked_unusable:
    # #     temp[landlocked] = [255, 0, 255, 255]
    # img = Image.fromarray(temp)
    # with open("meme_landlocked.png", "wb") as file:
    #     img.save(file)

    # # Show unusable
    # for unusable in unusable:
    #     array[unusable] = [255, 0, 255, 255]
    # img = Image.fromarray(array)
    # with open("stego_unusable.png", "wb") as file:
    #     img.save(file)

    # return sorted(set(indexes) - unusable)
    temp = array.copy()
    # temp[unusable] = [255, 0, 255, 255]
    temp[unusable] = [255, 0, 255]  # bmp
    # for landlocked in landlocked_unusable:
    #     temp[landlocked] = [255, 0, 255, 255]
    img = Image.fromarray(temp)
    with open("meme_landlocked.png", "wb") as file:
        img.save(file)
    return list(numpy.argwhere(numpy.ones(array.shape[:2], dtype=bool) & ~unusable))


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


if __name__ == '__main__':
    # print(len(coords_in_distance((10, 10), 2)))
    import time
    START = time.time()
    # find_editable_pixels(numpy.array(Image.open("meme.png")))
    find_editable_pixels(numpy.array(Image.open("castle.bmp")))
    print(f"Needed time: {time.time() - START}")
    sys.exit()
