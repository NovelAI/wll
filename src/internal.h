/* Copyright 2026 Anlatan Inc.
 * SPDX-License-Identifier: Apache-2.0 */

/* Shared internal definitions of the encoder. */
#ifndef WLL_INTERNAL_H
#define WLL_INTERNAL_H

#include <stdint.h>

#include "bitwriter.h"
#include "huffman.h"
#include "mem.h"

/* Builds a function for AVX2 as well as the baseline target and picks one at
 * load time (GCC/Clang function multiversioning, needs ifunc support). Used
 * on loops the compiler vectorizes. */
#if (defined(__x86_64__) || defined(__i386__)) && defined(__linux__) && \
    defined(__GNUC__)
#define WLL_TARGET_CLONES __attribute__((target_clones("avx2", "default")))
#else
#define WLL_TARGET_CLONES
#endif

/* Backward reference stream: what the entropy coder emits per position. */
enum { WLL_REF_LITERAL = 0, WLL_REF_CACHE = 1, WLL_REF_COPY = 2 };

typedef struct {
  uint32_t v;   /* literal: ARGB; cache: index; copy: distance (plane) code */
  uint16_t len; /* copy: length in pixels (1..4096); otherwise 1 */
  uint8_t kind;
} wll_ref;

typedef struct {
  wll_ref* refs;
  size_t size;
  size_t cap;
  wll_mem* mem;
} wll_refs;

/* Symbol histograms of one prefix code group. */
typedef struct {
  uint32_t green[WLL_MAX_ALPHABET]; /* literal green, length prefix, cache */
  uint32_t red[WLL_NUM_LITERAL_CODES];
  uint32_t blue[WLL_NUM_LITERAL_CODES];
  uint32_t alpha[WLL_NUM_LITERAL_CODES];
  uint32_t dist[WLL_NUM_DISTANCE_CODES];
} wll_histo;

/* Prefix coding of lengths and distance codes (value >= 1). */
static inline void wll_prefix_encode(uint32_t value, int* code, int* extra_bits,
                                     uint32_t* extra_value) {
  uint32_t v = value - 1;
  if (v < 2) {
    *code = (int)v;
    *extra_bits = 0;
    *extra_value = 0;
    return;
  }
  int hb = 31 - __builtin_clz(v);
  int second = (int)((v >> (hb - 1)) & 1);
  *extra_bits = hb - 1;
  *extra_value = v & ((1u << (hb - 1)) - 1);
  *code = 2 * hb + second;
}

static inline uint32_t wll_argb(uint8_t a, uint8_t r, uint8_t g, uint8_t b) {
  return ((uint32_t)a << 24) | ((uint32_t)r << 16) | ((uint32_t)g << 8) | b;
}

#endif /* WLL_INTERNAL_H */
