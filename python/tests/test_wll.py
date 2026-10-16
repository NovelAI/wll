# Copyright 2026 Anlatan Inc.
# SPDX-License-Identifier: Apache-2.0

import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import Image, features

import wll

pytestmark = pytest.mark.skipif(not features.check("webp"),
                                reason="Pillow cannot decode WebP")

LEVELS = ["default", "fast"]


def decode(data):
    with Image.open(io.BytesIO(bytes(data))) as im:
        assert im.format == "WEBP"
        return np.asarray(im.convert("RGBA"))


def sample(height=67, width=91, seed=0):
    """Smooth content, noise, and transparent pixels with junk RGB."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    img = np.empty((height, width, 4), np.uint8)
    img[..., 0] = x * 2
    img[..., 1] = y * 3
    img[..., 2] = rng.integers(0, 256, (height, width))
    img[..., 3] = np.where((x // 7 + y // 5) % 3 == 0, 0, 255)
    return img


@pytest.mark.parametrize("level", LEVELS)
def test_array_round_trip(level):
    img = sample()
    data = wll.encode(img, level=level)
    assert isinstance(data, memoryview) and data.readonly
    assert bytes(data[:4]) == b"RIFF" and bytes(data[8:12]) == b"WEBP"
    np.testing.assert_array_equal(decode(data), img)


@pytest.mark.parametrize("level", LEVELS)
def test_encoder_reuse(level):
    enc = wll.Encoder(level)
    for seed in range(3):
        img = sample(40 + seed, 30 + 7 * seed, seed)
        np.testing.assert_array_equal(decode(enc.encode(img)), img)


def test_strided_arrays():
    big = sample(80, 120)
    crop = big[10:70, 20:100]  # rows further apart: encoded in place
    np.testing.assert_array_equal(decode(wll.encode(crop)), crop)
    flipped = big[::-1]  # descending rows: copied first
    np.testing.assert_array_equal(decode(wll.encode(flipped)), flipped)
    every_other = big[:, ::2]  # pixels 8 bytes apart: copied first
    np.testing.assert_array_equal(decode(wll.encode(every_other)),
                                  every_other)


def test_buffer_protocol():
    img = sample()
    view = memoryview(img.tobytes()).cast("B", img.shape)
    np.testing.assert_array_equal(decode(wll.encode(view)), img)


def test_pil_images():
    img = sample()
    rgba = Image.fromarray(img, "RGBA")
    np.testing.assert_array_equal(decode(wll.encode(rgba)), img)
    rgb = Image.fromarray(np.ascontiguousarray(img[..., :3]), "RGB")
    expected = img.copy()
    expected[..., 3] = 255
    np.testing.assert_array_equal(decode(wll.encode(rgb)), expected)


HAS_ARROW = hasattr(Image.Image, "__arrow_c_array__")


def check_pil(im, expected, zero_copy):
    """Round trip of a PIL image; whether it goes through the Arrow export."""
    assert (wll._arrow_capsules(im) is not None) == (zero_copy and HAS_ARROW)
    np.testing.assert_array_equal(decode(wll.encode(im)), expected)


def test_pil_memory_layouts(tmp_path):
    img = sample()
    check_pil(Image.fromarray(img, "RGBA").copy(), img, True)
    path = tmp_path / "in.png"
    Image.fromarray(img, "RGBA").save(path)
    with Image.open(path) as im:  # decoded on demand
        check_pil(im, img, True)
    # Wrapping other memory: Pillow must not export these.
    check_pil(Image.fromarray(img, "RGBA"), img, False)
    check_pil(Image.frombuffer("RGBA", (img.shape[1], img.shape[0]),
                               img.tobytes(), "raw", "RGBA", 0, 1), img, False)
    path = tmp_path / "in.tif"
    Image.fromarray(img, "RGBA").save(path)  # uncompressed
    with Image.open(path) as im:  # memory-mapped file
        im.load()
        assert im.readonly
        check_pil(im, img, False)
    if hasattr(Image, "fromarrow"):
        src = Image.fromarray(img, "RGBA").copy()
        check_pil(Image.fromarrow(src, "RGBA", src.size), img, False)


def test_pil_large_image():
    # Larger than one of Pillow's memory blocks (16 MB by default).
    img = np.zeros((2100, 2100, 4), np.uint8)
    img[::7, ::3] = (1, 2, 3, 0)
    check_pil(Image.fromarray(img, "RGBA").copy(), img, False)


def test_write_to_file_object_and_path(tmp_path):
    img = sample()
    data = bytes(wll.encode(img))
    f = io.BytesIO()
    assert wll.encode(img, f) == len(data)
    assert f.getvalue() == data
    path = tmp_path / "out.webp"
    assert wll.encode(img, path) == len(data)
    assert path.read_bytes() == data
    with open(tmp_path / "raw.webp", "wb", buffering=0) as raw:
        assert wll.encode(img, raw) == len(data)
    assert (tmp_path / "raw.webp").read_bytes() == data


def test_threads():
    images = [sample(50 + i, 60, i) for i in range(16)]
    expected = [bytes(wll.Encoder().encode(img)) for img in images]
    with ThreadPoolExecutor(4) as pool:
        got = list(pool.map(lambda img: bytes(wll.encode(img)), images))
    assert got == expected


@pytest.mark.parametrize("level", LEVELS)
def test_decode_round_trip(level):
    img = sample()
    data = wll.encode(img, level=level)
    assert wll.info(data) == (img.shape[1], img.shape[0])
    got = wll.decode(data)
    assert got.dtype == np.uint8 and got.shape == img.shape
    np.testing.assert_array_equal(got, img)
    np.testing.assert_array_equal(wll.Decoder().decode(bytes(data)), img)


def test_decode_libwebp_files(tmp_path):
    img = sample(53, 61)
    for i, kwargs in enumerate(({"lossless": True, "exact": True},
                                {"lossless": True, "exact": True,
                                 "method": 6, "quality": 100},
                                {"lossless": True, "exact": True,
                                 "exif": b"Exif\0\0MM\0*"})):
        path = tmp_path / ("%d.webp" % i)
        Image.fromarray(img, "RGBA").save(path, **kwargs)
        np.testing.assert_array_equal(wll.decode(path), img)
        with open(path, "rb") as f:
            np.testing.assert_array_equal(wll.decode(f), img)
    # A palette image.
    few = sample()
    few[...] = few[0, 0]
    few[::3, ::2] = (1, 2, 3, 4)
    path = tmp_path / "palette.webp"
    Image.fromarray(few, "RGBA").save(path, lossless=True, exact=True)
    np.testing.assert_array_equal(wll.decode(path), few)


def test_decode_into():
    img = sample()
    data = wll.encode(img)
    big = np.zeros((img.shape[0] + 4, img.shape[1] + 10, 4), np.uint8)
    view = big[2:-2, 3:-7]  # rows further apart
    assert wll.decode(data, out=view) is view
    np.testing.assert_array_equal(view, img)
    assert not big[:2].any() and not big[:, :3].any()
    with pytest.raises(ValueError):
        wll.decode(data, out=np.zeros((img.shape[0], img.shape[1] + 1, 4),
                                      np.uint8))
    with pytest.raises(ValueError):
        wll.decode(data, out=np.zeros(img.shape, np.uint8)[:, ::-1])


def test_decode_errors(tmp_path):
    data = bytes(wll.encode(sample()))
    with pytest.raises(ValueError):
        wll.decode(data[:len(data) // 2])
    with pytest.raises(ValueError):
        wll.decode(b"not a webp file")
    path = tmp_path / "lossy.webp"
    Image.fromarray(sample(), "RGBA").save(path, quality=80)
    with pytest.raises(wll.UnsupportedError):
        wll.decode(path)


def test_decode_max_pixels():
    img = sample()
    data = wll.encode(img)
    pixels = img.shape[0] * img.shape[1]
    np.testing.assert_array_equal(wll.decode(data, max_pixels=pixels), img)
    with pytest.raises(ValueError, match="max_pixels"):
        wll.decode(data, max_pixels=pixels - 1)
    # A bare VP8L header declaring 16384x16384 is refused before allocating.
    header = bytes([0x2f, 0xff, 0xff, 0xff, 0x0f])
    assert wll.info(header) == (16384, 16384)
    with pytest.raises(ValueError, match="max_pixels"):
        wll.decode(header, max_pixels=1 << 24)


def test_decode_threads():
    datas = [bytes(wll.encode(sample(50 + i, 60, i))) for i in range(16)]
    expected = [wll.Decoder().decode(d) for d in datas]
    with ThreadPoolExecutor(4) as pool:
        got = list(pool.map(wll.decode, datas))
    for a, b in zip(got, expected):
        np.testing.assert_array_equal(a, b)


def test_invalid_input():
    with pytest.raises(TypeError):
        wll.encode(sample().astype(np.uint16))
    with pytest.raises(ValueError):
        wll.encode(np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(ValueError):
        wll.encode(np.zeros((0, 4, 4), np.uint8))
    with pytest.raises(ValueError):
        wll.encode(np.zeros((1, wll.MAX_DIMENSION + 1, 4), np.uint8))
    with pytest.raises(ValueError):
        wll.encode(memoryview(bytes(48)).cast("I"))  # 1-D, not uint8
    with pytest.raises(ValueError):
        wll.Encoder("slow")
