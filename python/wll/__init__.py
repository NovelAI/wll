"""Exact lossless WebP encoding and decoding of RGBA images.

All four channels are kept exactly, including the RGB values of fully
transparent pixels.

    import wll
    data = wll.encode(image)                    # memoryview of the .webp file
    wll.encode(image, "out.webp", level="fast")  # or a path or file object
    pixels = wll.decode(data)                   # (height, width, 4) uint8

`image` is a PIL image or a (height, width, 4) uint8 array (numpy, or any
object with the buffer protocol). Arrays are encoded without copying them if
their pixels are 4 bytes apart and rows ascend in memory (C-contiguous
arrays, and crops of them); others are copied first. PIL images are
converted with `Image.convert("RGBA")` unless already RGBA. With Pillow
11.2.1 or later, images that Pillow allocated itself in one memory block
(up to 16 MB by default) are encoded without copying them through Pillow's
Arrow export; other PIL images are copied once with `tobytes()`.

decode() reads lossless WebP still images (from wll, libwebp and others):
bytes-like data, a path or a binary file object. It returns a numpy array,
or a memoryview of the same shape without numpy, or decodes into `out`.
Lossy and animated files raise UnsupportedError.

The encoder and decoder release the GIL, so threads work in parallel.
"""
import os
import sys
import threading
from typing import Any, BinaryIO, Optional, Tuple, Union

from . import _wll
from ._wll import MAX_DIMENSION, UnsupportedError

__all__ = ["Decoder", "Encoder", "UnsupportedError", "decode", "encode",
           "info", "MAX_DIMENSION"]

_LEVELS = {"default": _wll.DEFAULT, "fast": _wll.FAST}

Destination = Union[str, bytes, "os.PathLike[Any]", BinaryIO]
Source = Union[bytes, bytearray, memoryview, str, "os.PathLike[Any]",
               BinaryIO]


def _level(level: str) -> int:
    try:
        return _LEVELS[level]
    except (KeyError, TypeError):
        raise ValueError(
            "level must be 'default' or 'fast', not %r" % (level,)) from None


def _arrow_capsules(image: Any) -> Optional[Tuple[Any, Any]]:
    """The pixels of an RGBA PIL image as Arrow PyCapsules, or None.

    Pillow (11.2.1 and later) exports images held in one memory block
    without copying them. Images that wrap other memory are left out:
    frombuffer()/fromarray() and memory-mapped files (Image.readonly), and
    fromarrow() (ImagingCore.readonly). Pillow 12.3 crashes when exporting
    those.
    """
    if not hasattr(image, "__arrow_c_array__"):
        return None
    image.load()
    if image.readonly or getattr(image.im, "readonly", 1):
        return None
    try:
        return image.__arrow_c_array__()
    except ValueError:  # held in several memory blocks (large images)
        return None


def _encode(enc: Any, image: Any) -> memoryview:
    pil = sys.modules.get("PIL.Image")
    if pil is not None and isinstance(image, pil.Image):
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        width, height = image.size
        capsules = _arrow_capsules(image)
        if capsules is not None:
            return enc.encode_arrow(capsules[0], capsules[1], width, height)
        return enc.encode(
            memoryview(image.tobytes()).cast("B", (height, width, 4)))
    return enc.encode(_array(image))


def _array(image: Any) -> Any:
    """An array as a (height, width, 4) uint8 buffer the encoder accepts."""
    np = sys.modules.get("numpy")
    if np is not None and isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            raise TypeError("array dtype must be uint8, not %s" % image.dtype)
        if image.ndim != 3 or image.shape[2] != 4:
            raise ValueError("array shape must be (height, width, 4), not %s"
                             % (image.shape,))
        strides = image.strides
        if strides[1:] != (4, 1) or strides[0] < 4 * image.shape[1]:
            image = np.ascontiguousarray(image)
    return image


def _write(fp: Destination, data: memoryview) -> int:
    if isinstance(fp, (str, bytes, os.PathLike)):
        with open(fp, "wb") as f:
            return _write(f, data)
    view = data
    while view:
        n = fp.write(view)
        if n is None:  # file objects that do not report a count write all
            break
        view = view[n:]
    return len(data)


class Encoder:
    """An encoder that keeps its work buffers between images.

    Use one per thread: an encoder used by two threads at once raises
    RuntimeError. `level` is "default" (about the size of
    `cwebp -lossless -exact -z 2`, several times faster) or "fast" (about
    17% larger, about 25 times faster than "default").
    """

    __slots__ = ("_enc", "level")

    def __init__(self, level: str = "default") -> None:
        self._enc = _wll.Encoder(_level(level))
        self.level = level

    def encode(self, image: Any,
               fp: Optional[Destination] = None) -> Union[memoryview, int]:
        """Encodes `image` as a lossless WebP file.

        Without `fp`, returns a read-only memoryview of the file (call
        bytes() on it for a bytes object). With `fp` (a path or a binary
        file object), writes the file there and returns its size.
        """
        data = _encode(self._enc, image)
        if fp is None:
            return data
        return _write(fp, data)


def _read(source: Source) -> Any:
    """The file's bytes: `source` itself unless a path or file object."""
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as f:
            return f.read()
    if hasattr(source, "read"):
        return source.read()
    return source


def _new_pixels(width: int, height: int) -> Any:
    try:
        import numpy as np
    except ImportError:
        return memoryview(bytearray(width * height * 4)).cast(
            "B", (height, width, 4))
    return np.empty((height, width, 4), np.uint8)


def info(source: Source) -> Tuple[int, int]:
    """The (width, height) of a WebP file (data, path or file object)."""
    return _wll.info(_read(source))


class Decoder:
    """A decoder that keeps its work buffers between images.

    Use one per thread: a decoder used by two threads at once raises
    RuntimeError.
    """

    __slots__ = ("_dec",)

    def __init__(self) -> None:
        self._dec = _wll.Decoder()

    def decode(self, source: Source, out: Any = None, *,
               max_pixels: Optional[int] = None) -> Any:
        """Decodes a lossless WebP file to RGBA.

        `source` is the file's data (bytes-like), a path or a binary file
        object. Returns a (height, width, 4) uint8 numpy array (a memoryview
        of that shape without numpy), or decodes into `out`, a writable
        array of that shape whose pixels are 4 bytes apart, and returns it.
        Raises UnsupportedError for lossy or animated files and ValueError
        for corrupt ones, and for images of more than `max_pixels` pixels
        (checked before allocating anything). A file of a few bytes can
        declare up to MAX_DIMENSION squared pixels (1 GiB of RGBA); set
        `max_pixels` for untrusted input.
        """
        data = _read(source)
        width, height = _wll.info(data)
        if max_pixels is not None and width * height > max_pixels:
            raise ValueError("image of %dx%d pixels exceeds max_pixels (%d)"
                             % (width, height, max_pixels))
        if out is None:
            out = _new_pixels(width, height)
        self._dec.decode_into(data, out)
        return out


_local = threading.local()


def encode(image: Any, fp: Optional[Destination] = None, *,
           level: str = "default") -> Union[memoryview, int]:
    """Encodes `image` with a per-thread Encoder; see Encoder.encode()."""
    encoders = getattr(_local, "encoders", None)
    if encoders is None:
        encoders = _local.encoders = {}
    enc = encoders.get(level)
    if enc is None:
        enc = encoders[level] = Encoder(level)
    return enc.encode(image, fp)


def decode(source: Source, out: Any = None, *,
           max_pixels: Optional[int] = None) -> Any:
    """Decodes with a per-thread Decoder; see Decoder.decode()."""
    dec = getattr(_local, "decoder", None)
    if dec is None:
        dec = _local.decoder = Decoder()
    return dec.decode(source, out, max_pixels=max_pixels)
