# wll: WebP lossless encoder and decoder

A small C encoder for lossless WebP (VP8L) that keeps 8-bit RGBA exactly,
including the RGB values of fully transparent pixels. It has no mode that
alters invisible pixels (libwebp's default without `-exact`). A decoder
reads its files and libwebp's lossless files about 1.5–1.9x faster than
libwebp.

There are two levels:

- `WLL_LEVEL_DEFAULT` aims at `cwebp -lossless -exact -z 2` file sizes and
  is about 2.6x faster than that.
- `WLL_LEVEL_FAST` makes about 17% larger files and is about 25x faster
  than the default level.

The encoder is tuned for NovelAI Diffusion outputs: photographic or painted
images with few exact repeats and rarely a small palette. It encodes on a
single thread; run one encoder per thread to encode many images.

## Results

The test set is 3079 NovelAI Diffusion images (3350.5 MP), lossless WebP
files from the image API; their sizes are the baseline ("API"). Sizes count the VP8L payload. All outputs were checked to
decode to the original RGBA.

| encoder | bpp | size vs API |
|---|---|---|
| API | 8.901 | 100.0% |
| cwebp -z 2 | 8.902 | 100.01% |
| wll | 8.865 | 99.59% |
| wll fast | 10.396 | 116.79% |

Per image, compared with `cwebp -lossless -exact -z 2`:

| encoder | median | p10 | p90 | worst |
|---|---|---|---|---|
| wll | -0.37% | -1.11% | -0.00% | +4.65% |
| wll fast | +16.83% | +12.13% | +22.63% | +40.86% |

The fast level's worst cases are flat, simple images (2.7–5 bpp with
cwebp). On those, long-distance matches and the color cache help, and the
fast level uses neither.

### Speed

Single-thread encode speed on a 200-image subset (213.5 MP). The test
machine is a Xeon Platinum 8562Y+ that was shared with other load, so
expect a few percent of noise.

| encoder | MP/s |
|---|---|
| cwebp -lossless -exact -z 2 | 3.81 |
| wll | 10.06 |
| wll fast | 237–267 |

The fast level measures 267 MP/s in `build/roundtrip-timing` and 237 MP/s in
`build/bench-png`, which runs other encoders in between. On a Threadripper
2950X (Zen 1), the default level runs at 8.13 MP/s (cwebp: 3.04), and the
fast level at about 199 MP/s on a 40-image subset.

### Compared with PNG encoders

Same 200-image subset and machine (`build/bench-png`):

| encoder | bpp | size vs wll | MP/s |
|---|---|---|---|
| wll | 8.803 | 100.0% | 10.06 |
| wll fast | 10.382 | 117.9% | 236.67 |
| libpng -1 | 14.604 | 165.9% | 14.71 |
| libpng -6 (default) | 13.047 | 148.2% | 3.36 |
| libpng -9 | 12.764 | 145.0% | 0.59 |
| fpnge 1 | 14.449 | 164.1% | 320.81 |
| fpnge 4 (default) | 13.578 | 154.2% | 264.89 |
| fpnge 5 | 13.568 | 154.1% | 245.05 |
| fpng | 15.053 | 171.0% | 184.24 |
| fpng slower | 14.090 | 160.1% | 114.13 |

### Compared with other WebP encoders

Lossless WebP encoders on public images from
[codec-corpus](https://github.com/imazen/codec-corpus), all opaque:

- photo: CLIC 2025, 62 images, 174.9 MP.
- photo512: CID22-512, 250 images, 65.5 MP. Mostly photos, a few charts.
- screenshot: the QOI benchmark's `screenshot_web` set, 13 images, 86.6 MP.

Each encoder ran on one pinned core of a KVM virtual machine with a Cascade
Lake Xeon (model and clock speed unknown), with all images loaded first and
the fastest of 2–5 runs kept. A repeat of the photo set for six of the
encoders gave speeds within 5%. Cascade Lake is older than the machine
above, so compare the speeds within this table rather than with the other
tables. libwebp 1.6.0 and zenwebp ran with `exact`; deepteams/webp with
`Exact`. libwebp 1.6.0 decoded every output except one (†), and each
matched its source byte for byte. oxideav-webp 0.3.1 and webpkit's default level are
not listed because some of their files did not decode in a 20-image trial.
Sizes are relative to libwebp `-z 2`, per set and over all 325 images:

| encoder | photo | photo512 | screenshot | all bpp | all | MP/s |
|---|---|---|---|---|---|---|
| wll fast | 110.7% | 113.7% | 124.8% | 7.111 | 112.4% | 167.93 |
| image-webp 0.2.4 | 118.1% | 122.7% | 133.9% | 7.613 | 120.4% | 103.02 |
| libwebp -z 0 | 107.9% | 111.3% | 175.1% | 7.151 | 113.1% | 23.06 |
| webpkit 0.1.0 (fast) | 226.1% | 206.5% | 554.7% | 15.287 | 241.7% | 18.20 |
| deepteams/webp 1.2.8 -z 0 | 210.1% | 209.0% | 249.8% | 13.430 | 212.3% | 15.24 |
| deepteams/webp 1.2.8 -z 1 | 143.2% | 135.6% | 168.1% | 9.029 | 142.7% | 10.48 |
| wll | 99.6% | 100.1% | 100.8% | 6.312 | 99.8% | 9.54 |
| libwebp -z 1 | 100.9% | 103.2% | 103.2% | 6.432 | 101.7% | 5.84 |
| deepteams/webp 1.2.8 -z 2 | 105.5% | 107.3% | 109.5% | 6.719 | 106.2% | 5.32 |
| zenwebp 0.4.4 -z 0 † | 101.8% | 104.6% | 104.4% | 6.497 | 102.7% | 4.76 |
| libwebp -z 2 | 100.0% | 100.0% | 100.0% | 6.325 | 100.0% | 4.18 |
| zenwebp 0.4.4 -z 1 | 100.9% | 103.3% | 103.5% | 6.435 | 101.7% | 4.10 |
| zenwebp 0.4.4 -z 2 | 100.1% | 100.0% | 100.1% | 6.330 | 100.1% | 3.04 |
| nativewebp 1.3.0 | 110.2% | 109.8% | 107.2% | 6.953 | 109.9% | 1.49 |

† One of zenwebp `-z 0`'s files is invalid: libwebp and wll both reject it.

On these images, the default level makes files as small as `-z 2` and is 2.3x
faster. The fast level is the fastest encoder measured, and its files are
about the size of `-z 0`'s (smaller on screenshots) at 7x the speed. Per
image, the default level is within +7.3% of `-z 2` (median +0.1%). The
fast level's worst images are flat graphics and screenshots (+69% to
+178%, under 1.1 bpp with `-z 2`), for the same reason as above.

Since every image is opaque, this comparison does not show whether an
encoder keeps the RGB values of transparent pixels.

### Decoding

Single-thread decode speed into a preallocated RGBA buffer, same machine,
200 images per set. The decoder's output was compared with libwebp's for
every image.

| files | libwebp | wll |
|---|---|---|
| API (libwebp's encoder) | 57.4 MP/s | 89.4 MP/s |
| wll default level | 58.4 MP/s | 89.0 MP/s |
| wll fast level | 58.5 MP/s | 110.8 MP/s |

On `cwebp -lossless -exact -z 4/6/9` files, which use a color cache (6
images each), wll decodes at 97–98 MP/s (libwebp: 65–68 MP/s). All 3079 API
files decode to the same pixels as with libwebp.

Compared with PNG decoders on the same 200 images, each format encoded
once (`build/bench-png-decode`):

| file | bpp | decoder | MP/s |
|---|---|---|---|
| PNG (libpng -6) | 13.047 | libpng | 50.50 |
|  |  | libspng | 51.70 |
|  |  | Wuffs | 75.46 |
| PNG (fpnge 4) | 13.578 | libpng | 44.47 |
|  |  | libspng | 45.39 |
|  |  | Wuffs | 66.49 |
| PNG (fpng) | 15.053 | libpng | 52.48 |
|  |  | libspng | 52.90 |
|  |  | Wuffs | 82.52 |
|  |  | fpng | 103.08 |
| WebP (wll) | 8.803 | libwebp | 58.28 |
|  |  | wll | 90.30 |
| WebP (wll fast) | 10.382 | libwebp | 58.49 |
|  |  | wll | 111.77 |

The PNG decoders verify checksums (zlib and chunk CRCs), as they do by
default; WebP has none. fpng's decoder only reads files that fpng wrote.

The decoder reads lossless WebP still images with all lossless features
(including palettes and color caches), in the simple or extended (VP8X)
format; metadata chunks are ignored. Lossy and animated files return
`WLL_ERR_UNSUPPORTED`.

### Memory

Peak memory one call adds, beyond its input and output buffers
(`build/memuse`, resident pages). Measured on an 832x1216 test image and
on a 2x2 tiling of it (1664x2432):

| operation | 832x1216 | 1664x2432 |
|---|---|---|
| wll encode | 36.0 MB (37 B/px) | 114.5 MB (30 B/px) |
| wll fast encode | 6.5 MB (6.7 B/px) | 23.5 MB (6.1 B/px) |
| libwebp encode (-z 2, exact) | 30.7 MB (32 B/px) | 114.7 MB (30 B/px) |
| wll decode | 4.2 MB (4.4 B/px) | 16.0 MB (4.2 B/px) |
| libwebp decode | 4.2 MB (4.3 B/px) | 8.8 MB (2.3 B/px) |

- The decoders need about the decoded image once more (4 bytes per
  pixel), since LZ77 copies refer back to it.
- libwebp's decoder keeps only a sliding window of rows: copies reach at
  most about a million pixels back. That saves memory from about 1 MP up.
- A `wll_encoder` or `wll_decoder` keeps its large buffers between calls
  for reuse, about the peak above. Measured after one call on the
  832x1216 image, an encoder keeps 35.5 MB (5.8 MB at the fast level) and
  a decoder 4.2 MB; on the 1664x2432 image 113.4 MB (21.0 MB) and 16.0 MB.
  The one-shot functions and libwebp free everything after each call.
  Python's `wll.encode()` and `wll.decode()` keep one encoder (per level)
  and one decoder per thread. At most 256 MB are kept per encoder or
  decoder (`wll_encoder_set_cache_limit()`, `wll_decoder_set_cache_limit()`);
  buffers beyond that are freed after each call, so a single huge image
  does not stay resident.

## How it works

**Default level.** It follows the structure of libwebp's method 2:

- Transforms: subtract green, a predictor per tile, and cross color.
- Matching: LZ77 with a hash chain, then a cost-based trace-back.
- Coding: clustered prefix code groups.

It differs from libwebp in a few places:

- It picks predictors with a log-magnitude cost proxy.
- The cross-color search is warm-started.
- The hash-chain window and iteration count are tuned for this content.
- Histogram clustering is incremental.

**Fast level.** It makes fixed choices instead of searching:

- Subtract green, and the "select" predictor (mode 11) for the whole image.
- No cross color.
- Copies only from the previous pixel or the pixel above.
- At most two prefix code groups. NAI images hide metadata in the alpha
  LSBs of the leftmost columns, so 32x32 tiles whose alpha residuals are
  all zero get their own group, where alpha costs no bits.

The fast level makes three passes:

1. Residuals and their equality bitmaps.
2. Copies found from the bitmaps, and symbol counts.
3. The bitstream.

On x86 both levels use AVX2 when the CPU has it (runtime dispatch), with
portable C fallbacks. The fast level writes literals with AVX2 gathers except
on Zen 1/2, where gathers are slow.

**Decoder.** Most of the time goes to prefix-code decoding:

- One lookup in a "pair table" decodes green and red together; constant
  alpha needs no lookup.
- Pixels are decoded in runs of tiles that share a code group, in a tight
  loop with the bit reader in registers. The refill happens off the
  critical path.
- With many groups, smaller tables keep more of them in the CPU caches.
- Inverse transforms run per row as soon as a row is decoded, with SSE2
  predictors, and adding green is fused into the RGBA output.

The input is untrusted. Every prefix code is checked before its table is
built, and all reads and copies are bounds-checked. `tools/fuzz_decode.c`
fuzzes the decoder and checks every file it accepts against libwebp;
`tools/fuzz_encode.c` round-trips arbitrary pixels through the encoder.

## API

```c
#include "wll.h"

wll_encoder* enc = wll_encoder_new();
wll_encoder_set_level(enc, WLL_LEVEL_FAST);  /* optional */
uint8_t* out;
size_t out_size;
if (wll_encoder_encode(enc, rgba, width, height, stride, &out, &out_size) ==
    WLL_OK) {
  /* out holds a complete .webp file */
  wll_free(out);
}
wll_encoder_free(enc);
```

`wll_encode_rgba()` is a one-shot call with the same arguments. An encoder
keeps its work buffers between calls; use one per thread. Images can be up
to 16384x16384.

```c
int width, height;
if (wll_get_info(data, size, &width, &height) == WLL_OK) {
  uint8_t* rgba = malloc((size_t)width * height * 4);
  wll_decoder* dec = wll_decoder_new();  /* reuse, one per thread */
  wll_status st = wll_decoder_decode(dec, data, size, rgba, width, height,
                                     (size_t)width * 4);
  wll_decoder_free(dec);
}
```

A file of other dimensions than `width` and `height` returns
`WLL_ERR_INVALID_ARG`. A file of a few bytes can declare 16384x16384 pixels (1 GiB of RGBA): with
untrusted input, check the size from `wll_get_info()` before allocating.
`wll_decode_rgba()` is a one-shot call that allocates the output.

## Python

```sh
pip install .            # needs GCC or Clang
```

```python
import wll

data = wll.encode(image)                     # read-only memoryview of the .webp
wll.encode(image, "out.webp", level="fast")  # or any binary file object
enc = wll.Encoder("fast")                    # explicit encoder, one per thread
data = enc.encode(image)

pixels = wll.decode(data)                    # (height, width, 4) uint8 array
pixels = wll.decode("in.webp")               # or a path or binary file object
wll.decode(data, out=array)                  # into an existing array
width, height = wll.info(data)
```

Decoding:

- `decode()` returns a numpy array, or a memoryview of the same shape when
  numpy is not installed. `Image.fromarray(pixels)` makes a PIL image of it.
- `out` can be any writable `(height, width, 4)` uint8 array whose pixels
  are 4 bytes apart, including crops of a larger array.
- Lossy and animated files raise `wll.UnsupportedError` (a `ValueError`);
  corrupt files raise `ValueError`.
- `max_pixels=` refuses larger images (`ValueError`) before anything is
  allocated. Set it for untrusted input: a file of a few bytes can declare
  16384x16384 pixels, 1 GiB of output.
- `wll.decode()` keeps one decoder per thread; `wll.Decoder()` is the
  explicit form. The GIL is released while decoding.

Encoding:

- `image` is a PIL image or a `(height, width, 4)` uint8 array: numpy, or
  any object with the buffer protocol.
- Arrays are encoded in place if their pixels are 4 bytes apart and their
  rows ascend in memory (C-contiguous arrays and crops of them). Other
  arrays are copied first.
- PIL images are converted with `convert("RGBA")` unless they are already
  RGBA.
- With Pillow 11.2.1 or later, images that Pillow allocated in one memory
  block (up to 16 MB by default) are encoded in place through Pillow's
  Arrow export. That covers decoded files, `new()`, `copy()` and
  `convert()` results.
- Other PIL images are copied once with `tobytes()`, which costs about 20%
  of the fast level's speed. These are older Pillow, larger images, and
  images wrapping other memory: `fromarray()`, `frombuffer()`, `fromarrow()`
  and memory-mapped files. Pillow 12.3 crashes when exporting those, so
  `wll` doesn't ask it to.
- The result is a memoryview of the encoder's output buffer, not a copy. Use
  `bytes(data)` when a bytes object is needed.
- With a destination, `encode()` writes the file there and returns its size.
- The GIL is released while encoding, so threads encode in parallel.
- `wll.encode()` keeps one encoder per thread and level; each encoder holds
  on to its work buffers.

Tests: `pip install .[test]` and `pytest python/tests`.

## Go

```sh
go get github.com/novelai/wll   # needs cgo and GCC or Clang
```

```go
import "github.com/novelai/wll"

// pix holds non-premultiplied RGBA, rows stride bytes apart.
data, err := wll.Encode(pix, width, height, stride, wll.LevelDefault)
```

- `wll.LevelFast` and `wll.LevelDefault` are the two levels.
- Each call creates and frees its own encoder, so no memory stays allocated
  between calls. `Encode` is safe to call from many goroutines; each call
  runs on one thread.
- The buffer length, stride and dimensions are checked before C reads the
  pixels; `Encode` returns an error instead.
- The result is a Go byte slice holding the complete .webp file.
- The decoder is not bound yet.
- The `cgo_*.c` files at the root compile `src/*.c` into the package, since
  cgo only builds C files in the package directory. `make go-stubs`
  regenerates them after a source file is added or removed.

Tests: `go test .` (also with `-race`).

## Building and testing

```sh
make            # build/wllenc, build/roundtrip
make test       # round trips (also under ASan/UBSan) through libwebp's decoder
build/wllenc [-fast] in.pam out.webp
```

The tests and benchmarks need a libwebp build (`LIBWEBP=...`, default
`third_party/libwebp`) to decode the output. Defining `WLL_DISABLE_AVX2`
builds the portable code paths only.

Tools:

- `build/roundtrip [-fast] *.pam`: checks the round trip and reports size
  and speed. Without arguments it runs the synthetic tests.
- `build/roundtrip-timing`: the same, with time per encoder stage.
- `tools/bench_full.py`: compares sizes with cwebp over a directory of
  images, in parallel, and checks exactness (`--wll-args=-fast` for the
  fast level).
- `tools/bench_cwebp.py --src DIR --pam DIR`: sizes and encode times of
  cwebp levels against a directory of WebP files, which it also decodes to
  PAM.
- `build/bench-png`: comparison with libpng, fpnge and fpng (`PNGSRC=...`,
  default `third_party/png`).
- `build/bench-decode *.webp`: decode speed against libwebp, with the
  output checked; `build/bench-decode-timing` adds time per stage.
- `build/bench-png-decode *.pam`: decode speed against libpng, libspng,
  Wuffs and fpng (`PNGSRC=...`, with Wuffs v0.3 and libspng checkouts).
- `build/memuse MODE file`: peak memory of one encode or decode, wll or
  libwebp.
- `build/fuzz-decode corpus_dir`: libFuzzer target (clang) for the decoder.
- `build/fuzz-encode corpus_dir`: libFuzzer target for the encoder, round
  trips of arbitrary pixels at both levels through wll and libwebp.

## License

Apache License 2.0 (`LICENSE`). Parts of the encoder and decoder are
derived from libwebp, under its BSD license; see `NOTICE`.
