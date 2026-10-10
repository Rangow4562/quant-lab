"""Squeezing integer codes into raw bytes: `bits` bits per code, no gaps."""
import numpy as np


def _offset(bits):
    return 2 ** (bits - 1) - 1


def pack_codes(codes, bits):
    """int8 codes in [-offset, offset] -> a uint8 array of tightly packed bits."""
    u = (codes.astype(np.int16).ravel() + _offset(bits)).astype(np.uint8)
    planes = ((u[:, None] >> np.arange(bits, dtype=np.uint8)) & 1).astype(np.uint8)
    return np.packbits(planes.ravel())


def pack_unsigned(codes, bits):
    """unsigned codes in [0, 2**bits - 1] -> a uint8 array of tightly packed bits."""
    u = codes.astype(np.int16).ravel().astype(np.uint8)
    planes = ((u[:, None] >> np.arange(bits, dtype=np.uint8)) & 1).astype(np.uint8)
    return np.packbits(planes.ravel())


def unpack_unsigned(buf, bits, shape):
    """The inverse of pack_unsigned, for codes of the given shape."""
    n = int(np.prod(shape))
    planes = np.unpackbits(buf)[:n * bits].reshape(n, bits)
    u = planes.dot(1 << np.arange(bits, dtype=np.uint16)).astype(np.int16)
    return u.astype(np.uint8 if bits <= 8 else np.int16).reshape(shape)


def unpack_codes(buf, bits, shape):
    """The inverse of pack_codes, for codes of the given shape."""
    n = int(np.prod(shape))
    planes = np.unpackbits(buf)[:n * bits].reshape(n, bits)
    u = planes.dot(1 << np.arange(bits, dtype=np.uint16)).astype(np.int16)
    return (u - _offset(bits)).astype(np.int8).reshape(shape)
