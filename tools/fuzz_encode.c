/* libFuzzer target for the encoder (build: make build/fuzz-encode, clang).
 *
 * The input picks the size, level and row padding and gives the pixels,
 * repeated to fill the image (repeats give the encoder copies to find).
 * Every image must encode without memory errors, and decode back exactly
 * with wll and with libwebp. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "webp/decode.h"
#include "wll.h"

int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  static wll_encoder* encs[2];
  static wll_decoder* dec;
  if (dec == NULL) {
    dec = wll_decoder_new();
    for (int l = 0; l < 2; ++l) {
      encs[l] = wll_encoder_new();
      wll_encoder_set_level(encs[l], l == 0 ? WLL_LEVEL_FAST
                                            : WLL_LEVEL_DEFAULT);
    }
  }
  if (size < 6) return 0;
  /* Up to 1024 x 256 or 256 x 1024. */
  const int tall = data[4] & 1;
  const int w = 1 + (tall ? data[0] : data[0] | (data[1] & 3) << 8);
  const int h = 1 + (tall ? data[1] | (data[0] & 3) << 8 : data[1]);
  const int level = (data[2] & 1) ? WLL_LEVEL_DEFAULT : WLL_LEVEL_FAST;
  const size_t stride = (size_t)w * 4 + (data[3] & 7);
  data += 5;
  size -= 5;
  uint8_t* rgba = (uint8_t*)malloc(stride * (size_t)h);
  for (int y = 0; y < h; ++y) {
    for (size_t x = 0; x < stride; ++x) {
      rgba[(size_t)y * stride + x] = data[((size_t)y * w * 4 + x) % size];
    }
  }
  uint8_t* out;
  size_t out_size;
  if (wll_encoder_encode(encs[level], rgba, w, h, stride, &out, &out_size) !=
      WLL_OK) {
    fprintf(stderr, "cannot encode %dx%d at level %d\n", w, h, level);
    abort();
  }
  const size_t row = (size_t)w * 4;
  uint8_t* own = (uint8_t*)malloc(row * (size_t)h);
  int lw, lh;
  uint8_t* ref = WebPDecodeRGBA(out, out_size, &lw, &lh);
  int ok = ref != NULL && lw == w && lh == h &&
           wll_decoder_decode(dec, out, out_size, own, w, h, row) == WLL_OK;
  for (int y = 0; ok && y < h; ++y) {
    const uint8_t* src = rgba + (size_t)y * stride;
    ok = memcmp(src, ref + (size_t)y * row, row) == 0 &&
         memcmp(src, own + (size_t)y * row, row) == 0;
  }
  if (!ok) {
    fprintf(stderr, "%dx%d at level %d does not round-trip\n", w, h, level);
    abort();
  }
  WebPFree(ref);
  free(own);
  wll_free(out);
  free(rgba);
  return 0;
}
