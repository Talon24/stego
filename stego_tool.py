"""Stego tool"""

import os  # pylint: disable=unused-import
import sys
# import cProfile
import getpass
import pathlib
import pdb  # pylint: disable=unused-import

import numpy
from PIL import Image
from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA

# from stego_basics import build_with_randomseed
from stego_basics import build_with_randomseed_short_header
from stego_basics import read_bytes_in_image
from stego_basics import keygen
from stego_basics import CleartextTooLarge
from stego_basics import build_stream
# from stego_basics import read_stream2
# from stego_basics import read_stream3
from stego_basics import read_stream4
from stego_basics import write_to_image2

from stego_basics import read_stream
from stego_basics import write_to_image
from stego_basics import write_to_image_avoid_clusters
# from stego_basics import
# from stego_basics import


def encrypt_symmetric(source, target, data_file, password):
    """main"""
    img = Image.open(source)
    key = keygen(password)
    # encrypt(b"Geheime Nachricht!", "passwort")
    # sys.exit()
    # print(dir(img))
    # print(img.size)
    width, height = img.size
    target_size = (width * height) // 8 * 2 * 3

    with open(data_file, "rb") as file:
        data = file.read()
    if len(data) > target_size:
        print("File too large!")
        raise CleartextTooLarge
    built = build_stream(data, key, target_size)
    read_stream(key, built)

    target_name = target
    write_to_image(built, img, target_name)


def encrypt_asymmetric(image_path, public_key_file, data_file,
                       target_name=None, avoid_clusters=True):
    """Hide data into an image using a public key."""
    image_path = pathlib.Path(image_path)
    img = Image.open(image_path)
    with open(public_key_file, "r") as file:
        public_key = file.read()
    rsakey = RSA.import_key(public_key, passphrase=None)
    key, randomseed = os.urandom(32), os.urandom(32)
    cipher_rsa = PKCS1_OAEP.new(rsakey)
    # encrypted_key = cipher_rsa.encrypt(key)
    # print(f"{hash(encrypted_key)=}")
    width, height = img.size
    target_size = (width * height) // 8 * 2 * 3

    with open(data_file, "rb") as file:
        data = file.read()
    if len(data) > target_size:
        print("File too large!")
        sys.exit(1)

    if target_name is None:
        target_name = f"{image_path.stem}_stego{image_path.suffix}"
    header, ciphertext = build_with_randomseed_short_header(
        data, key, randomseed, cipher_rsa)
    if avoid_clusters:
        write_to_image_avoid_clusters(header, ciphertext, img,
                                      randomseed, target_name)
    else:
        write_to_image2(header, ciphertext, img, randomseed, target_name)

    # TESTING
    # Decrypt
    # print("---------------------- Begin decryption")
    # print(decrypt_asymmetric(target_name, "testkey", "password",
    #                          avoid_clusters=avoid_clusters))


def decrypt_symmetric(filename, password):
    """read symmetrically encrypted data in stego image"""
    built = read_bytes_in_image(filename)
    key = keygen(password)
    return read_stream(key, built)


def decrypt_asymmetric(filename, keyfile, password=None, *,
                       interact=False, avoid_clusters=True):
    """read asymmetrically encrypted data in stego image."""
    if interact:
        password = getpass.getpass()
    with open(keyfile) as file:
        private_key = RSA.import_key(file.read(), passphrase=password)
    # built = read_bytes_in_image(filename, avoid_clusters=avoid_clusters)
    array = numpy.array(Image.open(filename))
    cipher_rsa = PKCS1_OAEP.new(private_key)
    # key = cipher_rsa.decrypt(built[:256])  # runtime: 0.015s on Desktop
    # return read_stream3(cipher_rsa, array, avoid_clusters=avoid_clusters)
    return read_stream4(cipher_rsa, array, avoid_clusters=avoid_clusters)


if __name__ == '__main__':
    # stego_encrypt("symmetric")
    # stego_encrypt_asymmetric()
    # decrypt_asymmetric("Nut_rsa.png", "testkey", "password")
    # mark("Nut.png")
    import time
    start = time.time()
    # encrypt_symmetric("meme.png", "meme_symmetric.png", "truncated_bee_movie.txt", "password")
    # print(decrypt_symmetric("meme_symmetric.png", "password"))
    encrypt_asymmetric("meme.png", "testkey.pub", "truncated_bee_movie.txt", avoid_clusters=True)
    print(decrypt_asymmetric("meme_stego.png", "testkey", "password", avoid_clusters=True))
    print(f"Finished in {time.time() - start:2.5}s")
    # import cProfile
    # cProfile.run("encrypt_asymmetric()")
