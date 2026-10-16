/* Copyright 2026 Anlatan Inc.
 * SPDX-License-Identifier: Apache-2.0 */

/* roundtrip: encode with wll, decode with libwebp, require identical RGBA.
 *
 * Without arguments, runs a set of synthetic images at both levels. With PAM
 * file arguments, tests those and reports size and speed; a leading -fast
 * selects WLL_LEVEL_FAST. */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pam.h"
#include "webp/decode.h"
#include "wll.h"

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return (uint32_t)(rng_state >> 32);
}

static double total_px, total_bytes, total_time;
/* Reused across images, as a batch encoder would; NULL for one-shot calls. */
static wll_encoder* g_enc;
static wll_decoder* g_dec;

/* Returns 1 if the image survives the round trip unchanged. */
static int check(const char* name, const uint8_t* rgba, int w, int h,
                 int verbose) {
  uint8_t* out;
  size_t out_size;
  double t0 = now();
  wll_status st =
      g_enc ? wll_encoder_encode(g_enc, rgba, w, h, (size_t)w * 4, &out,
                                 &out_size)
            : wll_encode_rgba(rgba, w, h, (size_t)w * 4, &out, &out_size);
  double t = now() - t0;
  if (st != WLL_OK) {
    printf("FAIL %s (%dx%d): encode status %d\n", name, w, h, (int)st);
    return 0;
  }
  int dw, dh;
  uint8_t* dec = WebPDecodeRGBA(out, out_size, &dw, &dh);
  int ok = dec != NULL && dw == w && dh == h &&
           memcmp(dec, rgba, (size_t)w * (size_t)h * 4) == 0;
  /* wll's own decoder, reused across images, must agree too. */
  if (ok) {
    uint8_t* own = (uint8_t*)malloc((size_t)w * (size_t)h * 4);
    wll_status ds = WLL_ERR_OUT_OF_MEMORY;
    if (own != NULL) {
      ds = wll_decoder_decode(g_dec, out, out_size, own, w, h, (size_t)w * 4);
    }
    if (ds != WLL_OK || memcmp(own, rgba, (size_t)w * (size_t)h * 4) != 0) {
      printf("FAIL %s (%dx%d): wll decoder %s\n", name, w, h,
             ds != WLL_OK ? "error" : "mismatch");
      if (ds == WLL_OK) {
        size_t i = 0;
        while (own[i] == rgba[i]) ++i;
        printf("     first mismatch at pixel (%zu,%zu) channel %zu: %d vs %d\n",
               (i / 4) % (size_t)w, (i / 4) / (size_t)w, i % 4, own[i],
               rgba[i]);
      } else {
        printf("     status %d\n", (int)ds);
      }
      free(own);
      WebPFree(dec);
      wll_free(out);
      return 0;
    }
    free(own);
  }
  /* The header's alpha hint must match the pixels. */
  WebPBitstreamFeatures features;
  int transparent = 0;
  for (size_t i = 3; i < (size_t)w * (size_t)h * 4; i += 4) {
    transparent |= rgba[i] != 255;
  }
  if (ok && (WebPGetFeatures(out, out_size, &features) != VP8_STATUS_OK ||
             features.has_alpha != transparent)) {
    printf("FAIL %s (%dx%d): alpha hint %d, image %s alpha\n", name, w, h,
           features.has_alpha, transparent ? "uses" : "has no");
    WebPFree(dec);
    wll_free(out);
    return 0;
  }
  if (!ok) {
    if (dec == NULL) {
      printf("FAIL %s (%dx%d): libwebp cannot decode\n", name, w, h);
    } else if (dw != w || dh != h) {
      printf("FAIL %s: decoded %dx%d, expected %dx%d\n", name, dw, dh, w, h);
    } else {
      size_t i = 0;
      while (dec[i] == rgba[i]) ++i;
      printf("FAIL %s (%dx%d): first mismatch at pixel (%zu,%zu) channel %zu\n",
             name, w, h, (i / 4) % (size_t)w, (i / 4) / (size_t)w, i % 4);
    }
  } else if (verbose) {
    printf("ok   %s %dx%d %zu bytes %.3f bpp %.2f MP/s\n", name, w, h, out_size,
           out_size * 8.0 / ((double)w * h), w * (double)h / t / 1e6);
  }
  total_px += (double)w * h;
  total_bytes += (double)out_size;
  total_time += t;
  WebPFree(dec);
  wll_free(out);
  return ok;
}

typedef enum {
  PAT_CONSTANT,
  PAT_NOISE,
  PAT_GRADIENT,
  PAT_TRANSPARENT_JUNK,
  PAT_FEW_COLORS,
  PAT_NOISY_ALPHA,
  PAT_SPARSE_NOISE,
  PAT_COUNT
} pattern;

static const char* kPatternNames[PAT_COUNT] = {
    "constant", "noise", "gradient", "transparent-junk", "few-colors",
    "noisy-alpha", "sparse-noise"};

static void fill(uint8_t* p, int w, int h, pattern pat) {
  uint32_t palette[5];
  for (int i = 0; i < 5; ++i) palette[i] = rnd();
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      uint8_t* q = p + ((size_t)y * w + x) * 4;
      uint32_t r = rnd();
      switch (pat) {
        case PAT_CONSTANT:
          q[0] = 12, q[1] = 34, q[2] = 56, q[3] = 255;
          break;
        case PAT_NOISE:
          memcpy(q, &r, 4);
          break;
        case PAT_GRADIENT:
          q[0] = (uint8_t)x, q[1] = (uint8_t)y, q[2] = (uint8_t)(x + y);
          q[3] = (uint8_t)(255 - (x ^ y));
          break;
        case PAT_TRANSPARENT_JUNK:
          /* Transparent pixels whose RGB must survive unchanged. */
          memcpy(q, &r, 3);
          q[3] = (x / 7 + y / 5) % 3 == 0 ? 0 : 255;
          break;
        case PAT_FEW_COLORS:
          memcpy(q, &palette[(x / 3 + y / 2) % 5], 4);
          break;
        case PAT_NOISY_ALPHA:
          q[0] = (uint8_t)(x * 3), q[1] = (uint8_t)(y * 5), q[2] = 7;
          q[3] = (uint8_t)(r & 1 ? 254 : 255);
          break;
        case PAT_SPARSE_NOISE:
          /* Small noise, and rare values that get the longest codes. */
          q[0] = (uint8_t)(100 + (r & 3));
          q[1] = (uint8_t)(100 + ((r >> 2) & 3));
          q[2] = (uint8_t)(100 + ((r >> 4) & 3));
          q[3] = 255;
          if ((r >> 8) % 1000 == 0) memcpy(q, &r, 3);
          break;
        default:
          break;
      }
    }
  }
}

static int run_synthetic(void) {
  static const int kSizes[][2] = {{1, 1},     {1, 7},   {7, 1},     {3, 3},
                                  {64, 64},   {257, 131}, {1000, 3}, {3, 1000},
                                  {16384, 2}, {2, 16384}, {640, 480}};
  int fails = 0, count = 0;
  for (size_t s = 0; s < sizeof(kSizes) / sizeof(kSizes[0]); ++s) {
    int w = kSizes[s][0], h = kSizes[s][1];
    uint8_t* p = (uint8_t*)malloc((size_t)w * h * 4);
    if (p == NULL) return 1;
    for (int pat = 0; pat < PAT_COUNT; ++pat) {
      char name[64];
      snprintf(name, sizeof(name), "%s", kPatternNames[pat]);
      fill(p, w, h, (pattern)pat);
      /* Exercise the one-shot call and the reused encoder at both levels. */
      wll_encoder* enc = g_enc;
      g_enc = NULL;
      fails += !check(name, p, w, h, 0);
      g_enc = enc;
      wll_encoder_set_level(g_enc, WLL_LEVEL_FAST);
      snprintf(name, sizeof(name), "%s (fast)", kPatternNames[pat]);
      fails += !check(name, p, w, h, 0);
      wll_encoder_set_level(g_enc, WLL_LEVEL_DEFAULT);
      count += 2;
    }
    free(p);
  }
  printf("synthetic: %d/%d passed\n", count - fails, count);
  return fails != 0;
}

int main(int argc, char** argv) {
  g_enc = wll_encoder_new();
  g_dec = wll_decoder_new();
  if (g_enc == NULL || g_dec == NULL) return 1;
  int first = 1;
  if (argc > 1 && strcmp(argv[1], "-fast") == 0) {
    wll_encoder_set_level(g_enc, WLL_LEVEL_FAST);
    first = 2;
  }
  if (argc <= first) {
    const int r = run_synthetic();
    wll_encoder_free(g_enc);
    wll_decoder_free(g_dec);
    return r;
  }
  int fails = 0;
  argc -= first - 1;
  argv += first - 1;
  for (int i = 1; i < argc; ++i) {
    int w, h;
    uint8_t* p = pam_read(argv[i], &w, &h);
    if (p == NULL) {
      printf("FAIL %s: cannot read\n", argv[i]);
      ++fails;
      continue;
    }
    fails += !check(argv[i], p, w, h, argc <= 6);
    free(p);
  }
  printf("%d/%d passed; %.1f MP, %.3f bpp, %.2f MP/s\n", argc - 1 - fails,
         argc - 1, total_px / 1e6, total_bytes * 8 / total_px,
         total_px / total_time / 1e6);
#ifdef WLL_TIMING
  {
    extern void wll_timing_report(void);
    wll_timing_report();
  }
#endif
  wll_encoder_free(g_enc);
  wll_decoder_free(g_dec);
  return fails != 0;
}
