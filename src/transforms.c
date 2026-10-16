/* Copyright 2026 Anlatan Inc.
 * Parts derived from libwebp, Copyright (c) 2010, Google Inc. (see NOTICE).
 * SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause */

#include "transforms.h"

#include "internal.h"

#include <stdlib.h>
#include <string.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

/* ------------------------------------------------------------------------ */
/* Pixel arithmetic (per channel, modulo 256 unless noted) */

static inline uint32_t sub_pixels(uint32_t a, uint32_t b) {
  uint32_t ag = 0x00ff00ffu + (a & 0xff00ff00u) - (b & 0xff00ff00u);
  uint32_t rb = 0xff00ff00u + (a & 0x00ff00ffu) - (b & 0x00ff00ffu);
  return (ag & 0xff00ff00u) | (rb & 0x00ff00ffu);
}

static inline uint32_t average2(uint32_t a, uint32_t b) {
  return (((a ^ b) & 0xfefefefeu) >> 1) + (a & b);
}

static inline uint32_t clip255(int v) {
  return v < 0 ? 0u : v > 255 ? 255u : (uint32_t)v;
}

static inline int ch(uint32_t v, int shift) { return (int)((v >> shift) & 0xff); }

static inline uint32_t clamped_add_subtract_full(uint32_t a, uint32_t b,
                                                 uint32_t c) {
  uint32_t r = 0;
  for (int s = 0; s < 32; s += 8) {
    r |= clip255(ch(a, s) + ch(b, s) - ch(c, s)) << s;
  }
  return r;
}

static inline uint32_t clamped_add_subtract_half(uint32_t c0, uint32_t c1,
                                                 uint32_t c2) {
  const uint32_t ave = average2(c0, c1);
  uint32_t r = 0;
  for (int s = 0; s < 32; s += 8) {
    int a = ch(ave, s);
    r |= clip255(a + (a - ch(c2, s)) / 2) << s;
  }
  return r;
}

/* Returns `a` (top) or `b` (left), whichever is closer to the gradient
 * estimate, with the decoder's tie-breaking. */
static inline uint32_t select_pred(uint32_t a, uint32_t b, uint32_t c) {
  int pa_minus_pb = 0;
  for (int s = 0; s < 32; s += 8) {
    pa_minus_pb += abs(ch(b, s) - ch(c, s)) - abs(ch(a, s) - ch(c, s));
  }
  return pa_minus_pb <= 0 ? a : b;
}

/* Prediction for an interior pixel (x > 0, y > 0); `top` points at the pixel
 * above. At the right edge top[1] is the first pixel of the current row. */
static inline uint32_t predict(int mode, uint32_t left, const uint32_t* top) {
  switch (mode) {
    case 0: return WLL_ARGB_BLACK;
    case 1: return left;
    case 2: return top[0];
    case 3: return top[1];
    case 4: return top[-1];
    case 5: return average2(average2(left, top[1]), top[0]);
    case 6: return average2(left, top[-1]);
    case 7: return average2(left, top[0]);
    case 8: return average2(top[-1], top[0]);
    case 9: return average2(top[0], top[1]);
    case 10: return average2(average2(left, top[-1]), average2(top[0], top[1]));
    case 11: return select_pred(top[0], left, top[-1]);
    case 12: return clamped_add_subtract_full(left, top[0], top[-1]);
    default: return clamped_add_subtract_half(left, top[0], top[-1]);
  }
}

/* Residuals of row `y`, columns [x0, x1), under predictor `mode`. */
WLL_TARGET_CLONES
static void residual_segment(int mode, const uint32_t* argb, int width, int x0,
                             int x1, int y, uint32_t* out) {
  const uint32_t* cur = argb + (size_t)y * (size_t)width;
  int x = x0;
  if (y == 0) {
    if (x == 0) {
      out[0] = sub_pixels(cur[0], WLL_ARGB_BLACK);
      ++x;
    }
    for (; x < x1; ++x) out[x - x0] = sub_pixels(cur[x], cur[x - 1]);
    return;
  }
  const uint32_t* up = cur - width;
  if (x == 0) {
    out[0] = sub_pixels(cur[0], up[0]);
    ++x;
  }
  /* Separate loops per mode so the switch is hoisted out of the pixel loop. */
#define LOOP(m)                                                    \
  case m:                                                          \
    for (; x < x1; ++x) {                                          \
      out[x - x0] = sub_pixels(cur[x], predict(m, cur[x - 1], up + x)); \
    }                                                              \
    break;
  switch (mode) {
    LOOP(0) LOOP(1) LOOP(2) LOOP(3) LOOP(4) LOOP(5) LOOP(6)
    LOOP(7) LOOP(8) LOOP(9) LOOP(10) LOOP(11) LOOP(12) LOOP(13)
    default: break;
  }
#undef LOOP
}

/* ------------------------------------------------------------------------ */

void wll_select_residual_row(const uint32_t* cur, const uint32_t* up,
                             int width, uint32_t* out) {
  if (up == NULL) {
    out[0] = sub_pixels(cur[0], WLL_ARGB_BLACK);
    for (int x = 1; x < width; ++x) out[x] = sub_pixels(cur[x], cur[x - 1]);
    return;
  }
  out[0] = sub_pixels(cur[0], up[0]);
  for (int x = 1; x < width; ++x) {
    out[x] = sub_pixels(cur[x], select_pred(up[x], cur[x - 1], up[x - 1]));
  }
}

void wll_subtract_green(uint32_t* argb, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    const uint32_t v = argb[i];
    const uint32_t g = (v >> 8) & 0xff;
    const uint32_t rb = ((v & 0x00ff00ffu) + 0x01000100u - (g << 16 | g)) &
                        0x00ff00ffu;
    argb[i] = (v & 0xff00ff00u) | rb;
  }
}

/* ------------------------------------------------------------------------ */
/* Predictor transform */

/* Returns 1 when some plane is constant with a value other than black's,
 * where mode 0 (black) would add residuals that the other modes avoid. */
static int black_mode_hurts(const uint32_t* argb, size_t n) {
  const uint32_t first = argb[0];
  uint32_t has_diff = 0;
  for (size_t i = 1; i < n; ++i) has_diff |= argb[i] ^ first;
  for (int p = 0; p < 4; ++p) {
    const uint32_t m = 0xff000000u >> (8 * p);
    if (!(has_diff & m) && (first & m) != (WLL_ARGB_BLACK & m)) return 1;
  }
  return 0;
}

/* The predictor of a tile is chosen by a cheap estimate of the residuals'
 * cost: the sum over all channels of floor(8 * log2(1 + |r|)) for each signed
 * byte residual r, with |r| >= 16 rounded to the middle of its 16-wide step.
 * On NovelAI images this picks modes at least as well as comparing the
 * entropy of each mode's residual histogram, at a fraction of the time. */
static const uint8_t kLogCostLow[16] = {0,  8,  12, 16, 18, 20, 22, 24,
                                        25, 26, 27, 28, 29, 30, 31, 32};
static const uint8_t kLogCostHigh[16] = {0,  37, 42, 46, 49, 51, 53, 55,
                                         56, 0,  0,  0,  0,  0,  0,  0};

static uint64_t residual_cost_c(const uint32_t* res, int n) {
  uint64_t s = 0;
  for (int i = 0; i < n; ++i) {
    for (int sh = 0; sh < 32; sh += 8) {
      int b = (int8_t)(res[i] >> sh);
      b = b < 0 ? -b : b;
      s += b < 16 ? kLogCostLow[b] : kLogCostHigh[b >> 4];
    }
  }
  return s;
}

#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("ssse3"))) static uint64_t residual_cost_ssse3(
    const uint32_t* res, int n) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i nibble = _mm_set1_epi8(0x0f);
  const __m128i low = _mm_loadu_si128((const __m128i*)kLogCostLow);
  const __m128i high = _mm_loadu_si128((const __m128i*)kLogCostHigh);
  __m128i acc = zero;
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    const __m128i v = _mm_loadu_si128((const __m128i*)(res + i));
    const __m128i a = _mm_abs_epi8(v); /* -128 becomes 128 as unsigned */
    const __m128i lo = _mm_and_si128(a, nibble);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(a, 4), nibble);
    const __m128i small = _mm_cmpeq_epi8(hi, zero);
    const __m128i c = _mm_or_si128(
        _mm_and_si128(small, _mm_shuffle_epi8(low, lo)),
        _mm_andnot_si128(small, _mm_shuffle_epi8(high, hi)));
    acc = _mm_add_epi64(acc, _mm_sad_epu8(c, zero));
  }
  uint64_t s = (uint64_t)_mm_cvtsi128_si64(acc) +
               (uint64_t)_mm_cvtsi128_si64(_mm_unpackhi_epi64(acc, acc));
  return s + residual_cost_c(res + i, n - i);
}
#endif

#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2"))) static uint64_t residual_cost_avx2(
    const uint32_t* res, int n) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i nibble = _mm256_set1_epi8(0x0f);
  const __m256i low = _mm256_broadcastsi128_si256(
      _mm_loadu_si128((const __m128i*)kLogCostLow));
  const __m256i high = _mm256_broadcastsi128_si256(
      _mm_loadu_si128((const __m128i*)kLogCostHigh));
  __m256i acc = zero;
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256i v = _mm256_loadu_si256((const __m256i*)(res + i));
    const __m256i a = _mm256_abs_epi8(v);
    const __m256i lo = _mm256_and_si256(a, nibble);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(a, 4), nibble);
    const __m256i small = _mm256_cmpeq_epi8(hi, zero);
    const __m256i c = _mm256_blendv_epi8(_mm256_shuffle_epi8(high, hi),
                                         _mm256_shuffle_epi8(low, lo), small);
    acc = _mm256_add_epi64(acc, _mm256_sad_epu8(c, zero));
  }
  const __m128i s128 = _mm_add_epi64(_mm256_castsi256_si128(acc),
                                     _mm256_extracti128_si256(acc, 1));
  const uint64_t s = (uint64_t)_mm_cvtsi128_si64(s128) +
                     (uint64_t)_mm_cvtsi128_si64(_mm_unpackhi_epi64(s128, s128));
  return s + residual_cost_c(res + i, n - i);
}
#endif

/* SIMD level for residual_cost(): 2: AVX2, 1: SSSE3, 0: neither. */
static int residual_cost_level(void) {
#if defined(__x86_64__) || defined(__i386__)
  return __builtin_cpu_supports("avx2")    ? 2
         : __builtin_cpu_supports("ssse3") ? 1
                                           : 0;
#else
  return 0;
#endif
}

static uint64_t residual_cost(const uint32_t* res, int n, int level) {
#if defined(__x86_64__) || defined(__i386__)
  if (level == 2) return residual_cost_avx2(res, n);
  if (level == 1) return residual_cost_ssse3(res, n);
#endif
  (void)level;
  return residual_cost_c(res, n);
}

int wll_predictor_transform(const uint32_t* argb, int width, int height,
                            int bits, uint32_t* modes, uint32_t* residuals) {
  const int tile_size = 1 << bits;
  const int tiles_x = wll_subsample_size(width, bits);
  const int tiles_y = wll_subsample_size(height, bits);
  const size_t tile_px = (size_t)tile_size * (size_t)tile_size;
  /* Residuals of the mode being tried and of the best one so far. */
  uint32_t* buf = (uint32_t*)malloc(2 * tile_px * sizeof(uint32_t));
  if (buf == NULL) return 0;
  uint32_t* cur_res = buf;
  uint32_t* best_res = buf + tile_px;
  const int min_mode =
      black_mode_hurts(argb, (size_t)width * (size_t)height) ? 1 : 0;
  const int simd = residual_cost_level();

  for (int ty = 0; ty < tiles_y; ++ty) {
    const int y0 = ty * tile_size;
    const int y1 = y0 + tile_size < height ? y0 + tile_size : height;
    for (int tx = 0; tx < tiles_x; ++tx) {
      const int x0 = tx * tile_size;
      const int x1 = x0 + tile_size < width ? x0 + tile_size : width;
      const int tw = x1 - x0;
      const int th = y1 - y0;
      int best_mode = min_mode;
      uint64_t best_cost = 0;
      for (int mode = min_mode; mode < WLL_NUM_PRED_MODES; ++mode) {
        for (int y = 0; y < th; ++y) {
          residual_segment(mode, argb, width, x0, x1, y0 + y,
                           cur_res + (size_t)y * (size_t)tw);
        }
        const uint64_t cost = residual_cost(cur_res, tw * th, simd);
        if (mode == min_mode || cost < best_cost) {
          best_cost = cost;
          best_mode = mode;
          uint32_t* tmp = best_res;
          best_res = cur_res;
          cur_res = tmp;
        }
      }
      modes[ty * tiles_x + tx] = WLL_ARGB_BLACK | ((uint32_t)best_mode << 8);
      for (int y = 0; y < th; ++y) {
        memcpy(residuals + (size_t)(y0 + y) * (size_t)width + x0,
               best_res + (size_t)y * (size_t)tw, (size_t)tw * sizeof(uint32_t));
      }
    }
  }
  free(buf);
  return 1;
}

/* ------------------------------------------------------------------------ */
/* Cross color transform */

typedef struct {
  int g2r, g2b, r2b; /* signed multipliers, 3.5 fixed point */
} multipliers;

static inline int color_delta(int pred, int color) {
  return ((int)(int8_t)pred * (int)(int8_t)color) >> 5;
}

static inline uint32_t transform_red(int g2r, uint32_t v) {
  return (uint32_t)(ch(v, 16) - color_delta(g2r, ch(v, 8))) & 0xff;
}

static inline uint32_t transform_blue(int g2b, int r2b, uint32_t v) {
  return (uint32_t)(ch(v, 0) - color_delta(g2b, ch(v, 8)) -
                    color_delta(r2b, ch(v, 16))) & 0xff;
}

/* One tile's distinct (green, red, blue) residuals and distinct (green, red)
 * pairs, each with its pixel count, plus the accumulated statistics that the
 * cost of each candidate is measured against. Residual tiles repeat values a
 * lot, so the candidates are evaluated over far fewer entries than pixels. */
typedef struct {
  int n3;
  int16_t *g3, *r3, *b3;
  uint16_t* c3;
  int n2;
  int16_t *g2, *r2;
  uint16_t* c2;
  uint8_t* tmp;
  const uint32_t* acc_red;
  const uint32_t* acc_blue;
  wll_acc_stats red_stats, blue_stats;
} cc_tile;

/* Open addressing table mapping keys to entry indices. Entries are valid
 * only while their stamp equals `gen`, so starting over is just ++gen. */
typedef struct {
  uint32_t* key;
  uint32_t* stamp;
  uint16_t* index;
  uint32_t mask;
  int shift;
  uint32_t gen;
} dedup_table;

/* Returns the index of `key`, or -1 after inserting it with index `next`. */
static inline int dedup_find(dedup_table* d, uint32_t key, int next) {
  uint32_t slot = (key * 0x9e3779b1u) >> d->shift;
  while (d->stamp[slot] == d->gen) {
    if (d->key[slot] == key) return d->index[slot];
    slot = (slot + 1) & d->mask;
  }
  d->stamp[slot] = d->gen;
  d->key[slot] = key;
  d->index[slot] = (uint16_t)next;
  return -1;
}

static double cross_color_cost(const wll_entropy_tab* t, const uint32_t* acc,
                               const wll_acc_stats* stats,
                               const uint32_t* histo) {
  return wll_combined_entropy(t, histo, acc, stats) +
         wll_prediction_bias(histo, 3.f, 240.f);
}

static void weighted_histogram(const uint8_t* v, const uint16_t* w, int n,
                               uint32_t histo[256]) {
  memset(histo, 0, 256 * sizeof(histo[0]));
  for (int i = 0; i < n; ++i) histo[v[i]] += w[i];
}

static double red_cost(const wll_entropy_tab* t, const cc_tile* ct,
                       multipliers prev_x, multipliers prev_y, int g2r) {
  uint32_t histo[256];
  const int m = (int8_t)g2r;
  const int n = ct->n2;
  const int16_t* restrict g = ct->g2;
  const int16_t* restrict r = ct->r2;
  uint8_t* restrict out = ct->tmp;
  for (int i = 0; i < n; ++i) out[i] = (uint8_t)(r[i] - ((m * g[i]) >> 5));
  weighted_histogram(out, ct->c2, n, histo);
  double cost = cross_color_cost(t, ct->acc_red, &ct->red_stats, histo);
  if ((g2r & 0xff) == (prev_x.g2r & 0xff)) cost -= 3.;
  if ((g2r & 0xff) == (prev_y.g2r & 0xff)) cost -= 3.;
  if (g2r == 0) cost -= 3.;
  return cost;
}

static double blue_cost(const wll_entropy_tab* t, const cc_tile* ct,
                        multipliers prev_x, multipliers prev_y, int g2b,
                        int r2b) {
  uint32_t histo[256];
  const int mg = (int8_t)g2b, mr = (int8_t)r2b;
  const int n = ct->n3;
  const int16_t* restrict g = ct->g3;
  const int16_t* restrict r = ct->r3;
  const int16_t* restrict b = ct->b3;
  uint8_t* restrict out = ct->tmp;
  for (int i = 0; i < n; ++i) {
    out[i] = (uint8_t)(b[i] - ((mg * g[i]) >> 5) - ((mr * r[i]) >> 5));
  }
  weighted_histogram(out, ct->c3, n, histo);
  double cost = cross_color_cost(t, ct->acc_blue, &ct->blue_stats, histo);
  if ((g2b & 0xff) == (prev_x.g2b & 0xff)) cost -= 3.;
  if ((g2b & 0xff) == (prev_y.g2b & 0xff)) cost -= 3.;
  if ((r2b & 0xff) == (prev_x.r2b & 0xff)) cost -= 3.;
  if ((r2b & 0xff) == (prev_y.r2b & 0xff)) cost -= 3.;
  if (g2b == 0) cost -= 3.;
  if (r2b == 0) cost -= 3.;
  return cost;
}

/* Multiplier search. Neighbouring tiles tend to use similar multipliers, so
 * each search starts from the best of zero and the left and above tiles'
 * values, then refines with steps of 16 down to 1 for red and 16 down to 2
 * for blue. libwebp instead starts from zero with steps of 32 and 16 and
 * stops at 4; the finer steps add about 0.3% on NovelAI images. Starting
 * with smaller steps saves time but can get stuck: up to 16% larger output
 * on some images. */
static const int kRedDeltas[] = {16, 8, 4, 2, 1};
static const int kBlueDeltas[] = {16, 8, 4, 2};
#define NUM_BLUE_DELTAS ((int)(sizeof(kBlueDeltas) / sizeof(kBlueDeltas[0])))

typedef struct {
  int g2b, r2b;
  double cost;
} blue_point;

/* Blue cost with memoization; the search revisits points. */
static double blue_cost_memo(const wll_entropy_tab* t, const cc_tile* ct,
                             multipliers prev_x, multipliers prev_y, int g2b,
                             int r2b, blue_point* seen, int* num_seen) {
  for (int s = 0; s < *num_seen; ++s) {
    if (seen[s].g2b == g2b && seen[s].r2b == r2b) return seen[s].cost;
  }
  const double c = blue_cost(t, ct, prev_x, prev_y, g2b, r2b);
  seen[*num_seen].g2b = g2b;
  seen[*num_seen].r2b = r2b;
  seen[(*num_seen)++].cost = c;
  return c;
}

static multipliers best_multipliers(const wll_entropy_tab* t, const cc_tile* ct,
                                    multipliers prev_x, multipliers prev_y) {
  multipliers best = {0, 0, 0};

  /* Green to red. */
  double best_cost = red_cost(t, ct, prev_x, prev_y, 0);
  {
    const int starts[2] = {prev_x.g2r, prev_y.g2r};
    for (int k = 0; k < 2; ++k) {
      if (starts[k] == 0 || (k == 1 && starts[1] == starts[0])) continue;
      const double c = red_cost(t, ct, prev_x, prev_y, starts[k]);
      if (c < best_cost) {
        best_cost = c;
        best.g2r = starts[k];
      }
    }
  }
  for (size_t i = 0; i < sizeof(kRedDeltas) / sizeof(kRedDeltas[0]); ++i) {
    const int delta = kRedDeltas[i];
    const int center = best.g2r;
    for (int off = -delta; off <= delta; off += 2 * delta) {
      const double c = red_cost(t, ct, prev_x, prev_y, center + off);
      if (c < best_cost) {
        best_cost = c;
        best.g2r = center + off;
      }
    }
  }

  /* Green and red to blue: axis and diagonal steps of shrinking size. */
  static const int kOffsets[8][2] = {{0, -1}, {0, 1},  {-1, 0}, {1, 0},
                                     {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
  blue_point seen[3 + NUM_BLUE_DELTAS * 8];
  int num_seen = 0;
  best_cost =
      blue_cost_memo(t, ct, prev_x, prev_y, 0, 0, seen, &num_seen);
  {
    const multipliers starts[2] = {prev_x, prev_y};
    for (int k = 0; k < 2; ++k) {
      const double c = blue_cost_memo(t, ct, prev_x, prev_y, starts[k].g2b,
                                      starts[k].r2b, seen, &num_seen);
      if (c < best_cost) {
        best_cost = c;
        best.g2b = starts[k].g2b;
        best.r2b = starts[k].r2b;
      }
    }
  }
  for (int i = 0; i < NUM_BLUE_DELTAS; ++i) {
    const int delta = kBlueDeltas[i];
    for (int a = 0; a < 8; ++a) {
      const int g2b = best.g2b + kOffsets[a][0] * delta;
      const int r2b = best.r2b + kOffsets[a][1] * delta;
      const double c =
          blue_cost_memo(t, ct, prev_x, prev_y, g2b, r2b, seen, &num_seen);
      if (c < best_cost) {
        best_cost = c;
        best.g2b = g2b;
        best.r2b = r2b;
      }
    }
  }
  return best;
}

static inline uint32_t multipliers_to_code(multipliers m) {
  return WLL_ARGB_BLACK | ((uint32_t)(m.r2b & 0xff) << 16) |
         ((uint32_t)(m.g2b & 0xff) << 8) | (uint32_t)(m.g2r & 0xff);
}

static inline multipliers code_to_multipliers(uint32_t c) {
  multipliers m = {(int8_t)(c & 0xff), (int8_t)((c >> 8) & 0xff),
                   (int8_t)((c >> 16) & 0xff)};
  return m;
}

int wll_cross_color_transform(const wll_entropy_tab* t, uint32_t* argb,
                              int width, int height, int bits,
                              uint32_t* codes) {
  const int tile_size = 1 << bits;
  const int tiles_x = wll_subsample_size(width, bits);
  const int tiles_y = wll_subsample_size(height, bits);
  const size_t tile_px = (size_t)tile_size * (size_t)tile_size;
  uint32_t acc_red[256] = {0};
  uint32_t acc_blue[256] = {0};
  multipliers prev_x = {0, 0, 0}, prev_y = {0, 0, 0};
  /* Entry counts are 16 bits. */
  if (tile_px > 65535) return 0;
  /* Hash table with at least twice as many slots as tile pixels. */
  int table_bits = 1;
  while (((size_t)1 << table_bits) < 2 * tile_px) ++table_bits;
  const size_t slots = (size_t)1 << table_bits;
  /* The hash table, then five int16 arrays and two counts per pixel, plus
   * the byte scratch. */
  uint8_t* buf = (uint8_t*)calloc(
      1, slots * (2 * sizeof(uint32_t) + sizeof(uint16_t)) +
             tile_px * (7 * sizeof(int16_t) + 1));
  if (buf == NULL) return 0;
  dedup_table dt;
  dt.key = (uint32_t*)buf;
  dt.stamp = dt.key + slots;
  dt.index = (uint16_t*)(dt.stamp + slots);
  dt.mask = (uint32_t)slots - 1;
  dt.shift = 32 - table_bits;
  dt.gen = 0;
  cc_tile ct;
  ct.g3 = (int16_t*)(dt.index + slots);
  ct.r3 = ct.g3 + tile_px;
  ct.b3 = ct.r3 + tile_px;
  ct.c3 = (uint16_t*)(ct.b3 + tile_px);
  ct.g2 = (int16_t*)(ct.c3 + tile_px);
  ct.r2 = ct.g2 + tile_px;
  ct.c2 = (uint16_t*)(ct.r2 + tile_px);
  ct.tmp = (uint8_t*)(ct.c2 + tile_px);
  ct.acc_red = acc_red;
  ct.acc_blue = acc_blue;

  for (int ty = 0; ty < tiles_y; ++ty) {
    const int y0 = ty * tile_size;
    const int y1 = y0 + tile_size < height ? y0 + tile_size : height;
    for (int tx = 0; tx < tiles_x; ++tx) {
      const int x0 = tx * tile_size;
      const int x1 = x0 + tile_size < width ? x0 + tile_size : width;
      if (ty > 0) prev_y = code_to_multipliers(codes[(ty - 1) * tiles_x + tx]);
      /* Distinct (green, red, blue) residuals... */
      ++dt.gen;
      ct.n3 = 0;
      for (int y = y0; y < y1; ++y) {
        const uint32_t* p = argb + (size_t)y * (size_t)width;
        for (int x = x0; x < x1; ++x) {
          const uint32_t key = p[x] & 0x00ffffffu;
          const int k = dedup_find(&dt, key, ct.n3);
          if (k >= 0) {
            ct.c3[k]++;
          } else {
            ct.g3[ct.n3] = (int8_t)(key >> 8);
            ct.r3[ct.n3] = (int8_t)(key >> 16);
            ct.b3[ct.n3] = (int8_t)key;
            ct.c3[ct.n3++] = 1;
          }
        }
      }
      /* ...and distinct (green, red) pairs. */
      ++dt.gen;
      ct.n2 = 0;
      for (int i = 0; i < ct.n3; ++i) {
        const uint32_t key = ((uint32_t)(uint8_t)ct.g3[i] << 8) |
                             (uint8_t)ct.r3[i];
        const int k = dedup_find(&dt, key, ct.n2);
        if (k >= 0) {
          ct.c2[k] += ct.c3[i];
        } else {
          ct.g2[ct.n2] = ct.g3[i];
          ct.r2[ct.n2] = ct.r3[i];
          ct.c2[ct.n2++] = ct.c3[i];
        }
      }
      wll_acc_stats_init(t, acc_red, &ct.red_stats);
      wll_acc_stats_init(t, acc_blue, &ct.blue_stats);
      prev_x = best_multipliers(t, &ct, prev_x, prev_y);
      codes[ty * tiles_x + tx] = multipliers_to_code(prev_x);

      for (int y = y0; y < y1; ++y) {
        uint32_t* p = argb + (size_t)y * (size_t)width;
        for (int x = x0; x < x1; ++x) {
          const uint32_t v = p[x];
          p[x] = (v & 0xff00ff00u) | (transform_red(prev_x.g2r, v) << 16) |
                 transform_blue(prev_x.g2b, prev_x.r2b, v);
        }
      }
      /* Accumulate statistics, skipping pixels that backward references are
       * likely to cover. */
      for (int y = y0; y < y1; ++y) {
        const size_t row = (size_t)y * (size_t)width;
        for (int x = x0; x < x1; ++x) {
          const size_t i = row + (size_t)x;
          const uint32_t v = argb[i];
          if (i >= 2 && v == argb[i - 2] && v == argb[i - 1]) continue;
          if (i >= (size_t)width + 2 && argb[i - 2] == argb[i - width - 2] &&
              argb[i - 1] == argb[i - width - 1] && v == argb[i - width]) {
            continue;
          }
          acc_red[(v >> 16) & 0xff]++;
          acc_blue[v & 0xff]++;
        }
      }
    }
  }
  free(buf);
  return 1;
}
