/* Copyright 2026 Anlatan Inc.
 * SPDX-License-Identifier: Apache-2.0 */

/* wllenc: encode an RGBA PAM file as lossless WebP. */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pam.h"
#include "wll.h"

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
  int level = WLL_LEVEL_DEFAULT;
  if (argc == 4 && strcmp(argv[1], "-fast") == 0) {
    level = WLL_LEVEL_FAST;
    ++argv;
    --argc;
  }
  if (argc != 3) {
    fprintf(stderr, "usage: %s [-fast] in.pam out.webp\n", argv[0]);
    return 2;
  }
  int w, h;
  uint8_t* rgba = pam_read(argv[1], &w, &h);
  if (rgba == NULL) {
    fprintf(stderr, "cannot read %s\n", argv[1]);
    return 1;
  }
  wll_encoder* enc = wll_encoder_new();
  if (enc == NULL || wll_encoder_set_level(enc, level) != WLL_OK) return 1;
  uint8_t* out;
  size_t out_size;
  double t0 = now();
  wll_status st =
      wll_encoder_encode(enc, rgba, w, h, (size_t)w * 4, &out, &out_size);
  double t = now() - t0;
  wll_encoder_free(enc);
  free(rgba);
  if (st != WLL_OK) {
    fprintf(stderr, "encoding failed: %d\n", (int)st);
    return 1;
  }
  FILE* f = fopen(argv[2], "wb");
  if (f == NULL || fwrite(out, 1, out_size, f) != out_size || fclose(f) != 0) {
    fprintf(stderr, "cannot write %s\n", argv[2]);
    return 1;
  }
  fprintf(stderr, "%dx%d: %zu bytes, %.3f bpp, %.3f s (%.2f MP/s)\n", w, h,
          out_size, out_size * 8.0 / ((double)w * h), t, w * (double)h / t / 1e6);
  wll_free(out);
  return 0;
}
