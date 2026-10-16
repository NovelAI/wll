/* libFuzzer target for the decoder (build: make build/fuzz-decode, clang).
 *
 * Any input must decode without memory errors. Against libwebp:
 * - whatever wll decodes, libwebp decodes to the same pixels;
 * - whatever libwebp decodes, wll decodes too, unless unsupported (lossy
 *   or animated files, the only WLL_ERR_UNSUPPORTED cases). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "webp/decode.h"
#include "wll.h"

int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  static wll_decoder* dec;
  if (dec == NULL) dec = wll_decoder_new();
  /* The output buffer follows wll's reading of the header. */
  int w = 0, h = 0, lw, lh;
  const int wll_header = wll_get_info(data, size, &w, &h) == WLL_OK;
  if (!wll_header && !WebPGetInfo(data, size, &w, &h)) return 0;
  if ((size_t)w * (size_t)h > ((size_t)1 << 22)) return 0;
  if (WebPGetInfo(data, size, &lw, &lh) &&
      (size_t)lw * (size_t)lh > ((size_t)1 << 22)) {
    return 0;
  }
  uint8_t* ref = WebPDecodeRGBA(data, size, &lw, &lh);
  /* Rows with some padding, varying with the input. */
  const size_t row = (size_t)w * 4;
  const size_t stride = row + 4 * (size & 3);
  uint8_t* own = (uint8_t*)malloc(stride * (size_t)(h - 1) + row);
  wll_status st = WLL_ERR_OUT_OF_MEMORY;
  if (own != NULL) {
    /* Without a cache limit, buffers are freed and reallocated every time. */
    wll_decoder_set_cache_limit(dec, (size & 16) ? 0 : WLL_DEFAULT_CACHE_LIMIT);
    if (wll_header) {
      /* Other dimensions than the file's are rejected before writing. */
      if (wll_decoder_decode(dec, data, size, own, w, h + 1, stride) !=
              WLL_ERR_INVALID_ARG ||
          wll_decoder_decode(dec, data, size, own, w + 1, h, stride + 4) !=
              WLL_ERR_INVALID_ARG) {
        fprintf(stderr, "wll accepts other dimensions than the file's\n");
        abort();
      }
      st = wll_decoder_decode(dec, data, size, own, w, h, stride);
    } else {
      st = wll_get_info(data, size, &w, &h);
    }
  }
  int same = st == WLL_OK && ref != NULL && lw == w && lh == h;
  for (int y = 0; same && y < h; ++y) {
    same = memcmp(ref + (size_t)y * row, own + (size_t)y * stride, row) == 0;
  }
  if (st == WLL_OK && !same) {
    fprintf(stderr, "wll decoded a file %s\n",
            ref == NULL ? "libwebp rejects" : "differently from libwebp");
    abort();
  }
  if (ref != NULL && own != NULL && st != WLL_OK &&
      st != WLL_ERR_UNSUPPORTED) {
    fprintf(stderr, "wll rejects (status %d) a file libwebp decodes\n",
            (int)st);
    abort();
  }
  WebPFree(ref);
  free(own);
  return 0;
}
