/* Copyright 2026 Anlatan Inc.
 * Parts derived from libwebp, Copyright (c) 2010, Google Inc. (see NOTICE).
 * SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause */

#include <stdlib.h>
#include <string.h>

#include "backward_refs.h"
#include "entropy.h"
#include "fast.h"
#include "histogram.h"
#include "internal.h"
#include "timing.h"
#include "transforms.h"
#include "wll.h"

#ifdef WLL_TIMING
#include <stdio.h>
double wll_timing[WLL_T_NUM];
void wll_timing_report(void) {
  static const char* kNames[WLL_T_NUM] = {
      "convert",    "analyze", "predictor", "cross-color", "hash chain",
      "parse",      "trace",   "cluster",   "emit",        "f:residual",
      "f:masks",    "f:rle",     "f:histo",   "f:emit",      "d:entropy",
      "d:rows"};
  double total = 0;
  for (int i = 0; i < WLL_T_NUM; ++i) total += wll_timing[i];
  for (int i = 0; i < WLL_T_NUM; ++i) {
    if (wll_timing[i] == 0) continue;
    fprintf(stderr, "%-12s %8.3f s %5.1f%%\n", kNames[i], wll_timing[i],
            100. * wll_timing[i] / total);
  }
}
#endif

#define VP8L_SIGNATURE 0x2f
#define VP8L_VERSION 0

enum {
  PREDICTOR_TRANSFORM = 0,
  CROSS_COLOR_TRANSFORM = 1,
  SUBTRACT_GREEN_TRANSFORM = 2,
};
#define MIN_TRANSFORM_BITS 2
#define MAX_TRANSFORM_BITS 9
#define MIN_HUFFMAN_BITS 2
#define MAX_HUFFMAN_BITS 9
#define MAX_HUFF_IMAGE_SIZE 2600
#define MAX_PREDICTOR_IMAGE_SIZE (1 << 14)

/* Picks the smallest bits >= `bits` whose subsampled image has at most
 * `image_size_max` pixels, without shrinking it to a single pixel needlessly. */
static int clamp_bits(int width, int height, int bits, int min_bits,
                      int max_bits, int image_size_max) {
  bits = bits < min_bits ? min_bits : bits > max_bits ? max_bits : bits;
  int size = wll_subsample_size(width, bits) * wll_subsample_size(height, bits);
  while (bits < max_bits && size > image_size_max) {
    ++bits;
    size = wll_subsample_size(width, bits) * wll_subsample_size(height, bits);
  }
  while (bits > min_bits && size == 1) {
    size = wll_subsample_size(width, bits - 1) *
           wll_subsample_size(height, bits - 1);
    if (size != 1) break;
    --bits;
  }
  return bits;
}

/* Hash chain search effort: candidates per pixel as libwebp sets them for
 * quality 25. */
#define HASH_CHAIN_ITERS 12
/* libwebp searches 16 rows back at quality 25; on NovelAI images 4 rows
 * compress as well and save time. */
#define HASH_CHAIN_WINDOW_ROWS 4

/* ------------------------------------------------------------------------ */
/* Entropy-coded image */

static void histo_add_refs(wll_histo* h, const wll_refs* r) {
  for (size_t i = 0; i < r->size; ++i) {
    const wll_ref* ref = &r->refs[i];
    if (ref->kind == WLL_REF_LITERAL) {
      h->green[(ref->v >> 8) & 0xff]++;
      h->red[(ref->v >> 16) & 0xff]++;
      h->blue[ref->v & 0xff]++;
      h->alpha[ref->v >> 24]++;
    } else if (ref->kind == WLL_REF_CACHE) {
      h->green[WLL_NUM_LITERAL_CODES + WLL_NUM_LENGTH_CODES + ref->v]++;
    } else {
      int code, nbits;
      uint32_t extra;
      wll_prefix_encode(ref->len, &code, &nbits, &extra);
      h->green[WLL_NUM_LITERAL_CODES + code]++;
      wll_prefix_encode(ref->v, &code, &nbits, &extra);
      h->dist[code]++;
    }
  }
}

typedef struct {
  wll_prefix_code green, red, blue, alpha, dist;
  uint8_t len_buf[WLL_MAX_ALPHABET + 3 * WLL_NUM_LITERAL_CODES +
                  WLL_NUM_DISTANCE_CODES];
  uint16_t code_buf[WLL_MAX_ALPHABET + 3 * WLL_NUM_LITERAL_CODES +
                    WLL_NUM_DISTANCE_CODES];
} code_group;

static void code_group_init(code_group* g, int cache_bits) {
  int sizes[5] = {
      WLL_NUM_LITERAL_CODES + WLL_NUM_LENGTH_CODES +
          (cache_bits > 0 ? 1 << cache_bits : 0),
      WLL_NUM_LITERAL_CODES, WLL_NUM_LITERAL_CODES, WLL_NUM_LITERAL_CODES,
      WLL_NUM_DISTANCE_CODES};
  wll_prefix_code* codes[5] = {&g->green, &g->red, &g->blue, &g->alpha,
                               &g->dist};
  size_t off = 0;
  for (int i = 0; i < 5; ++i) {
    codes[i]->n = sizes[i];
    codes[i]->len = g->len_buf + off;
    codes[i]->code = g->code_buf + off;
    off += (size_t)sizes[i];
  }
}

static void code_group_store(wll_bw* bw, const uint32_t* lit,
                             const uint32_t* red, const uint32_t* blue,
                             const uint32_t* alpha, const uint32_t* dist,
                             code_group* g) {
  wll_prefix_code_store(bw, lit, &g->green);
  wll_prefix_code_store(bw, red, &g->red);
  wll_prefix_code_store(bw, blue, &g->blue);
  wll_prefix_code_store(bw, alpha, &g->alpha);
  wll_prefix_code_store(bw, dist, &g->dist);
}

/* Emits `r` with the code group of the tile each reference starts in, as
 * given by `symbols` (tiles of 1 << bits); with symbols == NULL, all use
 * groups[0]. */
static void emit_refs(wll_bw* bw, const wll_refs* r, const code_group* groups,
                      const uint32_t* symbols, int xsize, int bits) {
  const int tiles_x = wll_subsample_size(xsize, bits);
  int x = 0, y = 0;
  for (size_t i = 0; i < r->size; ++i) {
    const wll_ref* ref = &r->refs[i];
    const code_group* g =
        symbols ? &groups[symbols[(y >> bits) * tiles_x + (x >> bits)]]
                : groups;
    x += ref->len;
    while (x >= xsize) {
      x -= xsize;
      ++y;
    }
    if (ref->kind == WLL_REF_LITERAL) {
      /* Codes are at most 15 bits, so pairs fit one write. */
      const uint32_t gs = (ref->v >> 8) & 0xff, rs = (ref->v >> 16) & 0xff;
      const uint32_t bs = ref->v & 0xff, as = ref->v >> 24;
      const int gl = g->green.len[gs], bl = g->blue.len[bs];
      wll_bw_put(bw, g->green.code[gs] | (uint32_t)g->red.code[rs] << gl,
                 gl + g->red.len[rs]);
      wll_bw_put(bw, g->blue.code[bs] | (uint32_t)g->alpha.code[as] << bl,
                 bl + g->alpha.len[as]);
    } else if (ref->kind == WLL_REF_CACHE) {
      wll_put_symbol(bw, &g->green,
                     WLL_NUM_LITERAL_CODES + WLL_NUM_LENGTH_CODES + (int)ref->v);
    } else {
      int code, nbits;
      uint32_t extra;
      wll_prefix_encode(ref->len, &code, &nbits, &extra);
      wll_put_symbol(bw, &g->green, WLL_NUM_LITERAL_CODES + code);
      wll_bw_put(bw, extra, nbits);
      wll_prefix_encode(ref->v, &code, &nbits, &extra);
      wll_put_symbol(bw, &g->dist, code);
      wll_bw_put(bw, extra, nbits);
    }
  }
}

/* Computes backward references for `argb` into `best`: a greedy hash chain
 * LZ77 parse, refined by a cost-based re-parse that uses the greedy parse's
 * statistics, keeping whichever is estimated smaller. (libwebp also tries an
 * RLE-only parse; it never wins against the re-parse, whose candidates
 * include the previous pixel and the pixel above.) Copy distances are left as
 * plane codes. Returns 0 on allocation failure. */
static int compute_refs(const wll_entropy_tab* t, const uint32_t* argb,
                        int width, int height, wll_refs* best, wll_mem* mem) {
  const size_t n = (size_t)width * (size_t)height;
  wll_hash_chain hc;
  wll_refs traced;
  int ok = 0;
  uint32_t window = (uint32_t)width * HASH_CHAIN_WINDOW_ROWS;
  if (window > WLL_WINDOW_SIZE) window = WLL_WINDOW_SIZE;
  if (!wll_hash_chain_init(&hc, n, mem)) return 0;
  if (!wll_refs_init(&traced, n, mem)) goto done;
  WLL_TIMER_START(timer);
  if (!wll_hash_chain_fill(&hc, argb, width, height, HASH_CHAIN_ITERS, window)) {
    goto done;
  }
  WLL_TIMER_LAP(timer, WLL_T_HASH_CHAIN);
  wll_refs_lz77(argb, width, height, &hc, best);
  const float bits_lz = wll_refs_estimate_bits(t, best, width);
  WLL_TIMER_LAP(timer, WLL_T_PARSE);
  if (!wll_refs_trace_backwards(argb, width, height, &hc, best, &traced,
                                mem)) {
    goto done;
  }
  const float bits_trace = wll_refs_estimate_bits(t, &traced, width);
  WLL_TIMER_LAP(timer, WLL_T_TRACE);
  if (bits_trace < bits_lz) {
    wll_ref* tmp = best->refs;
    best->refs = traced.refs;
    best->size = traced.size;
    traced.refs = tmp;
  }
  wll_refs_to_plane_codes(best, width);
  ok = 1;
done:
  wll_refs_free(&traced);
  wll_hash_chain_free(&hc);
  return ok;
}

static int encode_image_stream(wll_bw* bw, const wll_entropy_tab* t,
                               const uint32_t* argb, int width, int height,
                               int histo_bits);

/* Writes the meta prefix codes (if more than one group) and the code groups
 * of the main image, then its data. */
static int encode_grouped(wll_bw* bw, const wll_entropy_tab* t,
                          const wll_refs* refs, int width, int height,
                          int histo_bits) {
  const int tiles_x = wll_subsample_size(width, histo_bits);
  const int tiles_y = wll_subsample_size(height, histo_bits);
  const size_t tiles = (size_t)tiles_x * (size_t)tiles_y;
  wll_hist_set clusters = {0};
  code_group* groups = NULL;
  int* dense = NULL;
  int ok = 0;
  uint32_t* symbols = (uint32_t*)malloc(tiles * sizeof(*symbols));
  if (symbols == NULL) return 0;
  WLL_TIMER_START(timer);
  if (!wll_histogram_image(t, refs, width, height, histo_bits, 0, &clusters,
                           symbols, bw->mem)) {
    goto done;
  }
  WLL_TIMER_LAP(timer, WLL_T_CLUSTER);
  /* Number the clusters that are actually used densely. */
  dense = (int*)malloc(2 * (size_t)clusters.size * sizeof(*dense));
  if (dense == NULL) goto done;
  int* inverse = dense + clusters.size;
  for (int k = 0; k < clusters.size; ++k) dense[k] = -1;
  int num_groups = 0;
  for (size_t i = 0; i < tiles; ++i) {
    if (dense[symbols[i]] < 0) dense[symbols[i]] = num_groups++;
  }
  groups = (code_group*)wll_mem_alloc(bw->mem,
                                      (size_t)num_groups * sizeof(*groups));
  if (groups == NULL) goto done;

  if (num_groups > 1) {
    wll_bw_put(bw, 1, 1);
    wll_bw_put(bw, (uint32_t)(histo_bits - 2), 3);
    /* Group index in the green and red channels. */
    for (size_t i = 0; i < tiles; ++i) {
      symbols[i] = (uint32_t)dense[symbols[i]] << 8;
    }
    if (!encode_image_stream(bw, t, symbols, tiles_x, tiles_y, 0)) goto done;
    for (size_t i = 0; i < tiles; ++i) symbols[i] >>= 8;
  } else {
    wll_bw_put(bw, 0, 1);
  }
  /* Store the groups in index order. */
  for (int k = 0; k < clusters.size; ++k) {
    if (dense[k] >= 0) inverse[dense[k]] = k;
  }
  for (int d = 0; d < num_groups; ++d) {
    const wll_hist* h = clusters.h[inverse[d]];
    code_group_init(&groups[d], 0);
    code_group_store(bw, h->lit, h->red, h->blue, h->alpha, h->dist,
                     &groups[d]);
  }
  emit_refs(bw, refs, groups, num_groups > 1 ? symbols : NULL, width,
            histo_bits);
  WLL_TIMER_LAP(timer, WLL_T_EMIT);
  ok = 1;
done:
  wll_mem_free(bw->mem, groups);
  free(dense);
  free(symbols);
  wll_hist_set_free(&clusters);
  return ok;
}

/* Writes an entropy-coded image without color cache. For the main image,
 * `histo_bits` > 0 gives the tile size of its prefix code groups; sub-images
 * pass 0 and use a single group. Returns 0 on allocation failure. */
static int encode_image_stream(wll_bw* bw, const wll_entropy_tab* t,
                               const uint32_t* argb, int width, int height,
                               int histo_bits) {
  const size_t n = (size_t)width * (size_t)height;
  wll_refs refs = {NULL, 0, 0, NULL};
  wll_histo* h = NULL;
  code_group* g = NULL;
  int ok = 0;
  if (!wll_refs_init(&refs, n, bw->mem)) goto done;
  if (!compute_refs(t, argb, width, height, &refs, bw->mem)) goto done;

  wll_bw_put(bw, 0, 1); /* no color cache */
  if (histo_bits > 0) {
    ok = encode_grouped(bw, t, &refs, width, height, histo_bits);
    goto done;
  }
  h = (wll_histo*)calloc(1, sizeof(*h));
  g = (code_group*)malloc(sizeof(*g));
  if (h == NULL || g == NULL) goto done;
  histo_add_refs(h, &refs);
  code_group_init(g, 0);
  code_group_store(bw, h->green, h->red, h->blue, h->alpha, h->dist, g);
  emit_refs(bw, &refs, g, NULL, width, 0);
  ok = 1;
done:
  wll_refs_free(&refs);
  free(g);
  free(h);
  return ok;
}

/* ------------------------------------------------------------------------ */
/* Choice of transforms */

typedef enum {
  TX_DIRECT,
  TX_SPATIAL,
  TX_SUBGREEN,
  TX_SPATIAL_SUBGREEN,
  TX_NUM
} transform_set;

enum {
  H_ALPHA, H_RED, H_GREEN, H_BLUE,
  H_ALPHA_PRED, H_RED_PRED, H_GREEN_PRED, H_BLUE_PRED,
  H_RED_SUBGREEN, H_BLUE_SUBGREEN,
  H_RED_PRED_SUBGREEN, H_BLUE_PRED_SUBGREEN,
  H_NUM
};

static void add_channels(uint32_t p, uint32_t (*h)[256], int a, int r, int g,
                         int b) {
  h[a][p >> 24]++;
  h[r][(p >> 16) & 0xff]++;
  h[g][(p >> 8) & 0xff]++;
  h[b][p & 0xff]++;
}

static void add_subgreen(uint32_t p, uint32_t (*h)[256], int r, int b) {
  const uint32_t g = (p >> 8) & 0xff;
  h[r][((p >> 16) - g) & 0xff]++;
  h[b][(p - g) & 0xff]++;
}

/* Estimates which combination of subtract green and predictor transforms
 * compresses best, from histograms of pixels and of left-neighbour
 * differences (libwebp's AnalyzeEntropy, without palettes). Also reports
 * whether red and blue end up all zero, which makes cross color pointless.
 * Returns -1 on allocation failure. */
static int analyze_transforms(const wll_entropy_tab* t, const uint32_t* argb,
                              int width, int height, int transform_bits,
                              int* red_blue_zero) {
  uint32_t (*h)[256] = (uint32_t(*)[256])calloc(H_NUM, sizeof(*h));
  if (h == NULL) return -1;
  uint32_t prev = argb[0];
  /* All rows: sampling fewer flips close decisions, which can cost 10%. */
  for (int y = 0; y < height; ++y) {
    const uint32_t* row = argb + (size_t)y * (size_t)width;
    for (int x = 0; x < width; ++x) {
      const uint32_t p = row[x];
      const uint32_t ag = 0x00ff00ffu + (p & 0xff00ff00u) - (prev & 0xff00ff00u);
      const uint32_t rb = 0xff00ff00u + (p & 0x00ff00ffu) - (prev & 0x00ff00ffu);
      const uint32_t diff = (ag & 0xff00ff00u) | (rb & 0x00ff00ffu);
      prev = p;
      /* Repeats are left to backward references. */
      if (diff == 0 || (y > 0 && p == row[x - width])) continue;
      add_channels(p, h, H_ALPHA, H_RED, H_GREEN, H_BLUE);
      add_channels(diff, h, H_ALPHA_PRED, H_RED_PRED, H_GREEN_PRED, H_BLUE_PRED);
      add_subgreen(p, h, H_RED_SUBGREEN, H_BLUE_SUBGREEN);
      add_subgreen(diff, h, H_RED_PRED_SUBGREEN, H_BLUE_PRED_SUBGREEN);
    }
  }
  /* The zero difference is filtered out above, but likely occurs. */
  h[H_RED_PRED_SUBGREEN][0]++;
  h[H_BLUE_PRED_SUBGREEN][0]++;
  h[H_RED_PRED][0]++;
  h[H_GREEN_PRED][0]++;
  h[H_BLUE_PRED][0]++;
  h[H_ALPHA_PRED][0]++;

  double e[H_NUM];
  for (int i = 0; i < H_NUM; ++i) e[i] = wll_bits_entropy(t, h[i], 256);
  const double tiles = (double)wll_subsample_size(width, transform_bits) *
                       wll_subsample_size(height, transform_bits);
  double cost[TX_NUM];
  cost[TX_DIRECT] = e[H_ALPHA] + e[H_RED] + e[H_GREEN] + e[H_BLUE];
  cost[TX_SPATIAL] = e[H_ALPHA_PRED] + e[H_RED_PRED] + e[H_GREEN_PRED] +
                     e[H_BLUE_PRED] + tiles * log2(14.);
  cost[TX_SUBGREEN] = e[H_ALPHA] + e[H_RED_SUBGREEN] + e[H_GREEN] +
                      e[H_BLUE_SUBGREEN];
  cost[TX_SPATIAL_SUBGREEN] = e[H_ALPHA_PRED] + e[H_RED_PRED_SUBGREEN] +
                              e[H_GREEN_PRED] + e[H_BLUE_PRED_SUBGREEN] +
                              tiles * log2(24.);
  int best = TX_DIRECT;
  for (int i = 1; i < TX_NUM; ++i) {
    if (cost[i] < cost[best]) best = i;
  }

  static const int kRedBlue[TX_NUM][2] = {
      {H_RED, H_BLUE},
      {H_RED_PRED, H_BLUE_PRED},
      {H_RED_SUBGREEN, H_BLUE_SUBGREEN},
      {H_RED_PRED_SUBGREEN, H_BLUE_PRED_SUBGREEN}};
  *red_blue_zero = 1;
  for (int i = 1; i < 256; ++i) {
    if (h[kRedBlue[best][0]][i] | h[kRedBlue[best][1]][i]) {
      *red_blue_zero = 0;
      break;
    }
  }
  free(h);
  return best;
}

/* ------------------------------------------------------------------------ */
/* Container */

static void put_le32(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
}

static wll_status wrap_riff(const uint8_t* vp8l, size_t vp8l_size,
                            uint8_t** out, size_t* out_size) {
  const size_t pad = vp8l_size & 1;
  /* The RIFF size field is 32 bits. */
  if (vp8l_size > 0xffffffffu - 12 - pad) return WLL_ERR_INVALID_ARG;
  const size_t total = 12 + 8 + vp8l_size + pad;
  uint8_t* buf = (uint8_t*)malloc(total);
  if (buf == NULL) return WLL_ERR_OUT_OF_MEMORY;
  memcpy(buf, "RIFF", 4);
  put_le32(buf + 4, (uint32_t)(total - 8));
  memcpy(buf + 8, "WEBPVP8L", 8);
  put_le32(buf + 16, (uint32_t)vp8l_size);
  memcpy(buf + 20, vp8l, vp8l_size);
  if (pad) buf[20 + vp8l_size] = 0;
  *out = buf;
  *out_size = total;
  return WLL_OK;
}

/* ------------------------------------------------------------------------ */

static void put_header(wll_bw* bw, int width, int height, int alpha_used) {
  wll_bw_put(bw, VP8L_SIGNATURE, 8);
  wll_bw_put(bw, (uint32_t)(width - 1), 14);
  wll_bw_put(bw, (uint32_t)(height - 1), 14);
  wll_bw_put(bw, (uint32_t)alpha_used, 1);
  wll_bw_put(bw, VP8L_VERSION, 3);
}

/* WLL_LEVEL_FAST. The arguments are already checked. */
static wll_status encode_fast(wll_mem* mem, const uint8_t* rgba, int width,
                              int height, size_t stride, uint8_t** out,
                              size_t* out_size) {
  wll_status status = WLL_ERR_OUT_OF_MEMORY;
  wll_bw bw;
  int alpha_used;
  if (!wll_bw_init(&bw, (size_t)width * (size_t)height * 2 + 1024, mem)) {
    goto done;
  }
  /* Whether alpha is used is only known at the end; patched below. */
  put_header(&bw, width, height, 0);
  if (!wll_encode_fast(&bw, rgba, stride, width, height, mem, &alpha_used)) {
    goto done;
  }
  wll_bw_flush(&bw);
  if (bw.error) goto done;
  if (alpha_used) bw.buf[4] |= 0x10; /* bit 36 of the stream */
  status = wrap_riff(bw.buf, bw.pos, out, out_size);
done:
  wll_bw_free(&bw);
  return status;
}

static wll_status encode(wll_mem* mem, int level, const uint8_t* rgba,
                         int width, int height, size_t stride, uint8_t** out,
                         size_t* out_size) {
  if (out == NULL || out_size == NULL) return WLL_ERR_INVALID_ARG;
  *out = NULL;
  *out_size = 0;
  if (rgba == NULL || width < 1 || height < 1 || width > WLL_MAX_DIMENSION ||
      height > WLL_MAX_DIMENSION || stride < (size_t)width * 4) {
    return WLL_ERR_INVALID_ARG;
  }

  const size_t n = (size_t)width * (size_t)height;
  if (level == WLL_LEVEL_FAST) {
    return encode_fast(mem, rgba, width, height, stride, out, out_size);
  }
  WLL_TIMER_START(convert_timer);
  uint32_t* argb = (uint32_t*)wll_mem_alloc(mem, n * sizeof(*argb));
  if (argb == NULL) return WLL_ERR_OUT_OF_MEMORY;
  uint32_t alpha_and = 0xffffffffu;
  for (int y = 0; y < height; ++y) {
    const uint8_t* src = rgba + (size_t)y * stride;
    uint32_t* dst = argb + (size_t)y * (size_t)width;
    for (int x = 0; x < width; ++x) {
      dst[x] = wll_argb(src[4 * x + 3], src[4 * x], src[4 * x + 1],
                        src[4 * x + 2]);
      alpha_and &= dst[x];
    }
  }
  const int alpha_used = (alpha_and >> 24) != 0xff;
  WLL_TIMER_LAP(convert_timer, WLL_T_CONVERT);

  /* Tile sizes as libwebp picks them for method 2 without a palette. */
  const int histo_bits = clamp_bits(width, height, 5, MIN_HUFFMAN_BITS,
                                    MAX_HUFFMAN_BITS, MAX_HUFF_IMAGE_SIZE);
  const int transform_bits = histo_bits < 6 ? histo_bits : 6;
  const int pred_bits =
      clamp_bits(width, height, transform_bits, MIN_TRANSFORM_BITS,
                 MAX_TRANSFORM_BITS, MAX_PREDICTOR_IMAGE_SIZE);
  const int cc_bits = transform_bits;
  const int pred_w = wll_subsample_size(width, pred_bits);
  const int pred_h = wll_subsample_size(height, pred_bits);
  const int cc_w = wll_subsample_size(width, cc_bits);
  const int cc_h = wll_subsample_size(height, cc_bits);

  wll_status status = WLL_ERR_OUT_OF_MEMORY;
  wll_bw bw;
  const wll_entropy_tab* tab = NULL;
  uint32_t* residuals = NULL;
  uint32_t* modes = NULL;
  uint32_t* cc_codes = NULL;
  if (!wll_bw_init(&bw, n * 2 + 1024, mem)) goto done;

  put_header(&bw, width, height, alpha_used);

  tab = wll_entropy_tab_get();
  residuals = (uint32_t*)wll_mem_alloc(mem, n * sizeof(*residuals));
  modes = (uint32_t*)malloc((size_t)pred_w * pred_h * sizeof(*modes));
  cc_codes = (uint32_t*)malloc((size_t)cc_w * cc_h * sizeof(*cc_codes));
  if (residuals == NULL || modes == NULL || cc_codes == NULL) goto done;

  int red_blue_zero;
  WLL_TIMER_START(timer);
  const int tx = analyze_transforms(tab, argb, width, height, transform_bits,
                                    &red_blue_zero);
  if (tx < 0) goto done;
  WLL_TIMER_LAP(timer, WLL_T_ANALYZE);
  const int use_subtract_green = tx == TX_SUBGREEN || tx == TX_SPATIAL_SUBGREEN;
  const int use_predictor = tx == TX_SPATIAL || tx == TX_SPATIAL_SUBGREEN;
  const int use_cross_color = use_predictor && !red_blue_zero;

  /* Transforms, in the order the encoder applies them; the decoder undoes
   * them in reverse. */
  if (use_subtract_green) {
    wll_subtract_green(argb, n);
    wll_bw_put(&bw, 1, 1);
    wll_bw_put(&bw, SUBTRACT_GREEN_TRANSFORM, 2);
  }
  const uint32_t* image = argb;
  if (use_predictor) {
    WLL_TIMER_LAP(timer, WLL_T_ANALYZE);
    if (!wll_predictor_transform(argb, width, height, pred_bits, modes,
                                 residuals)) {
      goto done;
    }
    WLL_TIMER_LAP(timer, WLL_T_PREDICTOR);
    wll_bw_put(&bw, 1, 1);
    wll_bw_put(&bw, PREDICTOR_TRANSFORM, 2);
    wll_bw_put(&bw, (uint32_t)(pred_bits - MIN_TRANSFORM_BITS), 3);
    if (!encode_image_stream(&bw, tab, modes, pred_w, pred_h, 0)) goto done;
    image = residuals;
  }
  if (use_cross_color) {
    WLL_TIMER_LAP(timer, WLL_T_ANALYZE);
    if (!wll_cross_color_transform(tab, residuals, width, height, cc_bits,
                                   cc_codes)) {
      goto done;
    }
    WLL_TIMER_LAP(timer, WLL_T_CROSS_COLOR);
    wll_bw_put(&bw, 1, 1);
    wll_bw_put(&bw, CROSS_COLOR_TRANSFORM, 2);
    wll_bw_put(&bw, (uint32_t)(cc_bits - MIN_TRANSFORM_BITS), 3);
    if (!encode_image_stream(&bw, tab, cc_codes, cc_w, cc_h, 0)) goto done;
  }
  wll_bw_put(&bw, 0, 1); /* end of transforms */
  if (!encode_image_stream(&bw, tab, image, width, height, histo_bits)) {
    goto done;
  }
  wll_bw_flush(&bw);
  if (bw.error) goto done;
  status = wrap_riff(bw.buf, bw.pos, out, out_size);

done:
  wll_bw_free(&bw);
  free(cc_codes);
  free(modes);
  wll_mem_free(mem, residuals);
  wll_mem_free(mem, argb);
  return status;
}

struct wll_encoder {
  wll_mem mem;
  int level;
};

wll_encoder* wll_encoder_new(void) {
  wll_encoder* enc = (wll_encoder*)calloc(1, sizeof(wll_encoder));
  if (enc != NULL) {
    enc->level = WLL_LEVEL_DEFAULT;
    wll_mem_init(&enc->mem, WLL_DEFAULT_CACHE_LIMIT);
  }
  return enc;
}

wll_status wll_encoder_set_cache_limit(wll_encoder* enc, size_t bytes) {
  if (enc == NULL) return WLL_ERR_INVALID_ARG;
  wll_mem_set_limit(&enc->mem, bytes);
  return WLL_OK;
}

wll_status wll_encoder_set_level(wll_encoder* enc, int level) {
  if (enc == NULL || (level != WLL_LEVEL_FAST && level != WLL_LEVEL_DEFAULT)) {
    return WLL_ERR_INVALID_ARG;
  }
  enc->level = level;
  return WLL_OK;
}

void wll_encoder_free(wll_encoder* enc) {
  if (enc == NULL) return;
  wll_mem_release(&enc->mem);
  free(enc);
}

wll_status wll_encoder_encode(wll_encoder* enc, const uint8_t* rgba, int width,
                              int height, size_t stride, uint8_t** out,
                              size_t* out_size) {
  if (enc == NULL) return WLL_ERR_INVALID_ARG;
  return encode(&enc->mem, enc->level, rgba, width, height, stride, out,
                out_size);
}

wll_status wll_encode_rgba(const uint8_t* rgba, int width, int height,
                           size_t stride, uint8_t** out, size_t* out_size) {
  /* Reuse within the call only. */
  wll_mem mem;
  wll_mem_init(&mem, SIZE_MAX);
  wll_status status = encode(&mem, WLL_LEVEL_DEFAULT, rgba, width, height,
                             stride, out, out_size);
  wll_mem_release(&mem);
  return status;
}

void wll_free(void* p) { free(p); }
