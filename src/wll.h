/* Copyright 2026 Anlatan Inc.
 * SPDX-License-Identifier: Apache-2.0 */

/* wll: WebP lossless encoder.
 *
 * Encodes 8-bit RGBA as a lossless WebP (VP8L) file. All four channels are
 * preserved exactly, including the RGB values of fully transparent pixels.
 * There is deliberately no mode that alters invisible pixels (libwebp's
 * default, !exact): callers rely on bit-exact output.
 */
#ifndef WLL_H
#define WLL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WLL_MAX_DIMENSION 16384

typedef enum {
  WLL_OK = 0,
  WLL_ERR_INVALID_ARG,
  WLL_ERR_OUT_OF_MEMORY,
  WLL_ERR_INVALID_DATA, /* decoding: corrupt or truncated file */
  WLL_ERR_UNSUPPORTED,  /* decoding: WebP features this decoder lacks */
} wll_status;

/* Encodes non-premultiplied RGBA pixels (byte order R, G, B, A; rows `stride`
 * bytes apart) as a complete WebP file. On success *out receives a buffer of
 * *out_size bytes that must be released with wll_free(). */
wll_status wll_encode_rgba(const uint8_t* rgba, int width, int height,
                           size_t stride, uint8_t** out, size_t* out_size);

void wll_free(void* p);

/* An encoder keeps its large work buffers between calls, which saves the
 * cost of setting up fresh memory for every image. Use one per thread; an
 * encoder must not be used by two threads at once. */
typedef struct wll_encoder wll_encoder;

wll_encoder* wll_encoder_new(void);
void wll_encoder_free(wll_encoder* enc);

/* Speed/size trade-off of an encoder. WLL_LEVEL_DEFAULT (the default) aims
 * at libwebp's `cwebp -lossless -z 2` file sizes; WLL_LEVEL_FAST encodes
 * about 25 times faster with fixed choices, at about 17% larger files. */
enum { WLL_LEVEL_FAST = 0, WLL_LEVEL_DEFAULT = 1 };
wll_status wll_encoder_set_level(wll_encoder* enc, int level);

/* Same as wll_encode_rgba(), reusing the encoder's buffers. */
wll_status wll_encoder_encode(wll_encoder* enc, const uint8_t* rgba, int width,
                              int height, size_t stride, uint8_t** out,
                              size_t* out_size);

/* Decoding.
 *
 * The decoder reads lossless WebP still images, with all lossless features
 * and in the simple or extended (VP8X) format; metadata chunks are
 * ignored. Lossy and animated files return WLL_ERR_UNSUPPORTED. Corrupt
 * input returns WLL_ERR_INVALID_DATA; any input is safe to pass. */

/* Reads the dimensions of a WebP file. */
wll_status wll_get_info(const uint8_t* data, size_t size, int* width,
                        int* height);

/* Decodes a WebP file into a new RGBA buffer (width * 4 bytes per row),
 * which must be released with wll_free(). Images can declare up to
 * WLL_MAX_DIMENSION squared pixels (1 GiB of RGBA) in a few bytes; check
 * the size with wll_get_info() first to limit that. */
wll_status wll_decode_rgba(const uint8_t* data, size_t size, uint8_t** rgba,
                           int* width, int* height);

/* A decoder keeps work buffers between calls. Use one per thread. */
typedef struct wll_decoder wll_decoder;

wll_decoder* wll_decoder_new(void);
void wll_decoder_free(wll_decoder* dec);

/* Decodes into `rgba`, which holds `height` rows of `stride` >= width * 4
 * bytes. `width` and `height` are the image's (see wll_get_info()); a file
 * of other dimensions returns WLL_ERR_INVALID_ARG. The decoder's own buffers take about as much memory
 * as the image in `rgba`, plus an amount that grows with `size`. On failure
 * the contents of `rgba` are unspecified. */
wll_status wll_decoder_decode(wll_decoder* dec, const uint8_t* data,
                              size_t size, uint8_t* rgba, int width,
                              int height, size_t stride);

/* Encoders and decoders keep freed work buffers for reuse up to this many
 * bytes in total; larger buffers are released after each call. 0 keeps
 * none. */
#define WLL_DEFAULT_CACHE_LIMIT ((size_t)256 << 20)
wll_status wll_encoder_set_cache_limit(wll_encoder* enc, size_t bytes);
wll_status wll_decoder_set_cache_limit(wll_decoder* dec, size_t bytes);

#ifdef __cplusplus
}
#endif

#endif /* WLL_H */
