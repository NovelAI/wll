/* Copyright 2026 Anlatan Inc.
 * SPDX-License-Identifier: Apache-2.0 */

/* bench_decode: decoding speed of WebP files, libwebp against wll.
 *
 * Usage: bench_decode [-r rounds] file.webp...
 * Each file is decoded `rounds` times into a preallocated RGBA buffer; the
 * fastest round per file counts. wll's output must equal libwebp's. */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "webp/decode.h"
#include "wll.h"

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint8_t* read_file(const char* path, size_t* size) {
  FILE* f = fopen(path, "rb");
  if (f == NULL) return NULL;
  fseek(f, 0, SEEK_END);
  const long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t* data = n > 0 ? (uint8_t*)malloc((size_t)n) : NULL;
  if (data != NULL && fread(data, 1, (size_t)n, f) != (size_t)n) {
    free(data);
    data = NULL;
  }
  fclose(f);
  *size = (size_t)n;
  return data;
}

int main(int argc, char** argv) {
  int rounds = 3, first = 1;
  if (argc > 2 && strcmp(argv[1], "-r") == 0) {
    rounds = atoi(argv[2]);
    first = 3;
  }
  if (first >= argc || rounds < 1) {
    fprintf(stderr, "usage: %s [-r rounds] file.webp...\n", argv[0]);
    return 2;
  }
  double pixels = 0, seconds = 0, bytes = 0, wll_seconds = 0;
  int wll_files = 0, unsupported = 0;
  wll_decoder* dec = wll_decoder_new();
  if (dec == NULL) return 1;
  for (int i = first; i < argc; ++i) {
    size_t size;
    uint8_t* data = read_file(argv[i], &size);
    int w, h;
    if (data == NULL || !WebPGetInfo(data, size, &w, &h)) {
      fprintf(stderr, "cannot read %s\n", argv[i]);
      return 1;
    }
    const size_t stride = (size_t)w * 4;
    uint8_t* rgba = (uint8_t*)malloc(stride * (size_t)h);
    double best = 1e30;
    for (int r = 0; r < rounds; ++r) {
      const double t0 = now();
      if (WebPDecodeRGBAInto(data, size, rgba, stride * (size_t)h,
                             (int)stride) == NULL) {
        fprintf(stderr, "cannot decode %s\n", argv[i]);
        return 1;
      }
      const double t = now() - t0;
      if (t < best) best = t;
    }
    /* wll, checked against libwebp's output. */
    uint8_t* own = (uint8_t*)malloc(stride * (size_t)h);
    double wll_best = 1e30;
    wll_status st = WLL_OK;
    for (int r = 0; r < rounds && st == WLL_OK; ++r) {
      const double t0 = now();
      st = wll_decoder_decode(dec, data, size, own, w, h, stride);
      const double t = now() - t0;
      if (t < wll_best) wll_best = t;
    }
    if (st == WLL_ERR_UNSUPPORTED) {
      ++unsupported;
    } else if (st != WLL_OK || memcmp(own, rgba, stride * (size_t)h) != 0) {
      fprintf(stderr, "wll %s %s\n", st != WLL_OK ? "fails on" : "differs on",
              argv[i]);
      return 1;
    } else {
      ++wll_files;
      wll_seconds += wll_best;
    }
    free(own);
    pixels += (double)w * h;
    bytes += (double)size;
    seconds += best;
    free(rgba);
    free(data);
  }
  printf("%d files, %.1f MP, %.3f bpp: libwebp %.2f MP/s", argc - first,
         pixels / 1e6, bytes * 8 / pixels, pixels / seconds / 1e6);
  if (unsupported == 0) {
    printf(", wll %.2f MP/s\n", pixels / wll_seconds / 1e6);
  } else {
    printf("; wll: %d files unsupported, %d decoded identically\n",
           unsupported, wll_files);
  }
  wll_decoder_free(dec);
#ifdef WLL_TIMING
  {
    extern void wll_timing_report(void);
    wll_timing_report();
  }
#endif
  return 0;
}
