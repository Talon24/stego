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
PIXEL_IN_CLUSTER = 2000
PIXEL_NEXT_TO_CLUSTER = 1000


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


class BitString2():
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
    needed_header_pixels = needed_pixels(header)
    needed_message_pixels = needed_pixels(data)

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
    write_header(header, array)
    needed_header_pixels = needed_pixels(header)
    needed_message_pixels = needed_pixels(data)

    random.seed(randomseed)
    # print(f"{needed_header_pixels=}   ---- write_to_image2")
    unusables = find_editable_pixels(array)
    indexes = list(numpy.ndindex(array.shape[:2]))[needed_header_pixels:]
    indexes = set(indexes)
    indexes -= set(unusables)
    indexes = list(sorted(indexes))
    print(f"Hash of pixels found: {hash(tuple(indexes))}")
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


def read_stream2(key, cipher, array):  # pylint: disable=too-many-locals
    """Finds the header in the ciphertext and gives to array."""
    lennonce = cipher[:16]
    lencipher = cipher[16:48]
    randnonce = cipher[48:64]
    randcipher = cipher[64:96]
    nonce = cipher[96:112]
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
        all_pixels = find_editable_pixels(array)
    print(f"Hash of pixels found: {hash(tuple(all_pixels))}")
    # key = cipher_rsa.decrypt(built[:256])
    data = read_bytes_in_image2(array, all_pixels)
    ciphered_key, cipher = data[:256], data[256:]
    key = key = cipher_rsa.decrypt(ciphered_key)
    lennonce = cipher[:16]
    lencipher = cipher[16:48]
    randnonce = cipher[48:64]
    randcipher = cipher[64:96]
    nonce = cipher[96:112]
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
    length = len(indexes)
    for counter, position in enumerate(indexes):
        print(f"processing pixel {counter}", end="\r")
        if counter >= 1000:
            break
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
    pixel[0] = pixel[0] >> 2 << 2
    pixel[1] = pixel[1] >> 2 << 2
    pixel[2] = pixel[2] >> 2 << 2
    pixel[0] += value
    pixel[1] += value
    pixel[2] += value


def surrounding_pixels(array, coords):
    """Get all the surrounding pixels in an array."""
    shape = array.shape
    # print(shape[:1])
    new_positions = []
    first, second = coords[0], coords[1]
    for ypos in range(first - 1, first + 2):
        for xpos in range(second - 1, second + 2):
            # print(xpos, ypos)
            # if xpos == first and ypos == second:
            #     continue
            if xpos >= shape[0] or ypos >= shape[1] or xpos < 0 or ypos < 0:
                continue
            new_positions.append((ypos, xpos))
    return new_positions


def mark_cluster_unusable(unusable: set, landlocked_unusable: set,
                          position: tuple, array: numpy.array):
    """mark a cluster of a color as unusable"""
    initial_lsb = DataPixel(array[position]).low_value

    candidates = {position}
    # i = 0
    visited = set()
    while candidates:
        # print(f"depth: {i}", end="\r")
        # print(candidates)
        # i += 1
        current = candidates.copy()
        candidates = set()
        current -= visited
        for candidate in current:
            # if candidate in unusable:
            #     continue
            if DataPixel(array[candidate]).low_value == initial_lsb:
                unusable.add(candidate)
                visited.add(candidate)
                surroundings = surrounding_pixels(array, candidate)
                amount_pixels_same_color = 0
                for surrounding_pixel in surroundings:
                    # console_image.draw(array, unusable, surrounding_pixel)
                    if surrounding_pixel in unusable:
                        continue
                    amount_pixels_same_color += 1
                    # time.sleep(0.5)
                    if DataPixel(array[surrounding_pixel]).low_value == initial_lsb:
                        candidates.add(surrounding_pixel)
                        unusable.add(surrounding_pixel)
                if amount_pixels_same_color == len(surroundings):
                    # Surrounded totally by same-colored pixels
                    landlocked_unusable.add(candidate)
        # candidates = set(surroundings)
    # print()


def find_editable_pixels(array):
    """mark elements as writable or not"""
    unusable = set()
    landlocked_unusable = set()
    # queue = collections.deque(maxlen=array.shape[0])
    indexes = numpy.ndindex(array.shape[:2])
    for idx, position in enumerate(indexes):
        print(f"{idx / (array.shape[0] * array.shape[1]) * 100:6.2f}%", end="\r")
        if position in unusable:
            continue
        # import ipdb; ipdb.set_trace()
        surroundings = surrounding_pixels(array, position)
        current_lsb = DataPixel(array[position]).low_value
        # neighbors = []
        # for coord in surroundings:
        #     neighbors.append(DataPixel(array[coord]))
        neighbors = [DataPixel(array[coord]) for coord in surroundings]
        if len({neighbor.low_value for neighbor in neighbors}) == 1:
            mark_cluster_unusable(unusable, landlocked_unusable, position, array)
        # else:
            # import ipdb; ipdb.set_trace()
            # unusable.add(position)
            # for coord in surroundings:
            #     if DataPixel(array[coord]).low_value == current_lsb:
            #         unusable.add(coord)
    # print(len(unusable))
    print("\nFinished searching editable pixels.")
    border_pixels = unusable - landlocked_unusable
    # Make pixels next to clusters unusable too
    for position in border_pixels:
        for position in surrounding_pixels(array, position):
            unusable.add(position)
    return sorted(set(indexes) - unusable)
    # for position in unusable:
    #     array[position] = [0xff, 0x00, 0xff, 255]
    # image = Image.fromarray(array)
    # for position in unusable:
    #     image.putpixel(position, (0xff, 0x00, 0xff))
    # with open("unusable.png", "wb") as file:
    #     image.save(file)
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


if __name__ == '__main__':
    # print(len(coords_in_distance((10, 10), 2)))
    array_ = numpy.array(Image.open("meme.png"))
    find_editable_pixels(array_)
