/* Copyright 2026 Anlatan Inc.
 * Parts derived from libwebp, Copyright (c) 2010, Google Inc. (see NOTICE).
 * SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause */

#include "huffman.h"

#include <stdlib.h>
#include <string.h>

#define CODE_LENGTH_CODES 19
#define MAX_CODE_LENGTH_CODE_LENGTH 7

static const uint8_t kCodeLengthCodeOrder[CODE_LENGTH_CODES] = {
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

typedef struct {
  uint64_t weight;
  int sym;
} leaf;

static int leaf_cmp(const void* a, const void* b) {
  const leaf* x = (const leaf*)a;
  const leaf* y = (const leaf*)b;
  if (x->weight != y->weight) return x->weight < y->weight ? -1 : 1;
  return x->sym - y->sym;
}

void wll_huff_build(const uint32_t* counts, int n, int max_len, uint8_t* len) {
  leaf leaves[WLL_MAX_ALPHABET];
  uint64_t inner_weight[WLL_MAX_ALPHABET];
  int parent[2 * WLL_MAX_ALPHABET];
  uint8_t depth[2 * WLL_MAX_ALPHABET];
  int m = 0;
  assert(n <= WLL_MAX_ALPHABET);
  memset(len, 0, (size_t)n);
  for (int i = 0; i < n; ++i) {
    if (counts[i]) leaves[m++].sym = i;
  }
  if (m == 0) return;
  if (m == 1) {
    len[leaves[0].sym] = 1;
    return;
  }
  /* Raising small counts to count_min flattens the tree until it fits. */
  for (uint64_t count_min = 1;; count_min *= 2) {
    for (int i = 0; i < m; ++i) {
      uint64_t c = counts[leaves[i].sym];
      leaves[i].weight = c < count_min ? count_min : c;
    }
    qsort(leaves, (size_t)m, sizeof(leaves[0]), leaf_cmp);
    /* Two-queue merge. Nodes 0..m-1 are the sorted leaves, m..2m-2 the inner
     * nodes in creation order, so every parent index exceeds its children. */
    int li = 0, ii = 0, ni = 0;
    for (int k = 0; k < m - 1; ++k) {
      int pick[2];
      uint64_t w = 0;
      for (int j = 0; j < 2; ++j) {
        if (li < m && (ii >= ni || leaves[li].weight <= inner_weight[ii])) {
          w += leaves[li].weight;
          pick[j] = li++;
        } else {
          w += inner_weight[ii];
          pick[j] = m + ii++;
        }
      }
      parent[pick[0]] = parent[pick[1]] = m + ni;
      inner_weight[ni++] = w;
    }
    int root = 2 * m - 2;
    int max_depth = 0;
    depth[root] = 0;
    for (int i = root - 1; i >= 0; --i) {
      depth[i] = (uint8_t)(depth[parent[i]] + 1);
      if (i < m && depth[i] > max_depth) max_depth = depth[i];
    }
    if (max_depth <= max_len) {
      for (int i = 0; i < m; ++i) len[leaves[i].sym] = depth[i];
      return;
    }
  }
}

static uint32_t reverse_bits(uint32_t v, int n) {
  uint32_t r = 0;
  for (int i = 0; i < n; ++i) {
    r = (r << 1) | (v & 1);
    v >>= 1;
  }
  return r;
}

void wll_huff_canonical(const uint8_t* len, int n, uint16_t* code) {
  int bl_count[WLL_MAX_CODE_LENGTH + 1] = {0};
  uint32_t next[WLL_MAX_CODE_LENGTH + 1];
  for (int i = 0; i < n; ++i) bl_count[len[i]]++;
  bl_count[0] = 0;
  uint32_t c = 0;
  for (int b = 1; b <= WLL_MAX_CODE_LENGTH; ++b) {
    c = (c + (uint32_t)bl_count[b - 1]) << 1;
    next[b] = c;
  }
  for (int i = 0; i < n; ++i) {
    code[i] = len[i] ? (uint16_t)reverse_bits(next[len[i]]++, len[i]) : 0;
  }
}

typedef struct {
  uint8_t code;  /* 0..18 */
  uint8_t extra; /* extra bits value for 16/17/18 */
} token;

static token* repeated_zeros(token* t, int reps) {
  while (reps > 0) {
    if (reps < 3) {
      for (; reps > 0; --reps) *t++ = (token){0, 0};
    } else if (reps < 11) {
      *t++ = (token){17, (uint8_t)(reps - 3)};
      reps = 0;
    } else {
      int k = reps < 138 ? reps : 138;
      *t++ = (token){18, (uint8_t)(k - 11)};
      reps -= k;
    }
  }
  return t;
}

static token* repeated_values(token* t, int reps, int value, int prev) {
  if (value != prev) {
    *t++ = (token){(uint8_t)value, 0};
    --reps;
  }
  while (reps > 0) {
    if (reps < 3) {
      for (; reps > 0; --reps) *t++ = (token){(uint8_t)value, 0};
    } else {
      int k = reps < 6 ? reps : 6;
      *t++ = (token){16, (uint8_t)(k - 3)};
      reps -= k;
    }
  }
  return t;
}

/* Run-length codes `len` into code length code tokens. */
static int tokenize(const uint8_t* len, int n, token* tokens) {
  token* t = tokens;
  int prev = 8; /* decoder's initial "previous non-zero length" */
  for (int i = 0; i < n;) {
    int value = len[i];
    int k = i + 1;
    while (k < n && len[k] == value) ++k;
    if (value == 0) {
      t = repeated_zeros(t, k - i);
    } else {
      t = repeated_values(t, k - i, value, prev);
      prev = value;
    }
    i = k;
  }
  return (int)(t - tokens);
}

static int bits_log2_floor(uint32_t v) {
  int r = 0;
  while (v >>= 1) ++r;
  return r;
}

static void store_full_code(wll_bw* bw, const uint8_t* len, int n) {
  token tokens[WLL_MAX_ALPHABET];
  uint32_t histo[CODE_LENGTH_CODES] = {0};
  uint8_t cl_len[CODE_LENGTH_CODES];
  uint16_t cl_code[CODE_LENGTH_CODES];
  static const uint8_t kExtraBits[3] = {2, 3, 7};

  int num_tokens = tokenize(len, n, tokens);
  for (int i = 0; i < num_tokens; ++i) histo[tokens[i].code]++;
  wll_huff_build(histo, CODE_LENGTH_CODES, MAX_CODE_LENGTH_CODE_LENGTH, cl_len);
  wll_huff_canonical(cl_len, CODE_LENGTH_CODES, cl_code);

  int num_codes = CODE_LENGTH_CODES;
  while (num_codes > 4 && cl_len[kCodeLengthCodeOrder[num_codes - 1]] == 0) {
    --num_codes;
  }
  wll_bw_put(bw, (uint32_t)(num_codes - 4), 4);
  for (int i = 0; i < num_codes; ++i) {
    wll_bw_put(bw, cl_len[kCodeLengthCodeOrder[i]], 3);
  }

  /* A code with a single used symbol takes no bits per symbol. */
  int used = 0;
  for (int i = 0; i < CODE_LENGTH_CODES; ++i) used += cl_len[i] != 0;
  if (used == 1) memset(cl_len, 0, sizeof(cl_len));

  /* Optionally stop reading tokens before the trailing zero runs. */
  int trimmed = num_tokens;
  int trailing_bits = 0;
  for (int i = num_tokens - 1; i >= 0; --i) {
    int c = tokens[i].code;
    if (c != 0 && c != 17 && c != 18) break;
    --trimmed;
    trailing_bits += cl_len[c] + (c == 17 ? 3 : c == 18 ? 7 : 0);
  }
  int write_trimmed = trimmed > 1 && trailing_bits > 12;
  wll_bw_put(bw, (uint32_t)write_trimmed, 1);
  if (write_trimmed) {
    if (trimmed == 2) {
      wll_bw_put(bw, 0, 3 + 2);
    } else {
      int nbitpairs = bits_log2_floor((uint32_t)(trimmed - 2)) / 2 + 1;
      wll_bw_put(bw, (uint32_t)(nbitpairs - 1), 3);
      wll_bw_put(bw, (uint32_t)(trimmed - 2), nbitpairs * 2);
    }
    num_tokens = trimmed;
  }
  for (int i = 0; i < num_tokens; ++i) {
    int c = tokens[i].code;
    wll_bw_put(bw, cl_code[c], cl_len[c]);
    if (c >= 16) wll_bw_put(bw, tokens[i].extra, kExtraBits[c - 16]);
  }
}

void wll_prefix_code_store(wll_bw* bw, const uint32_t* counts,
                           wll_prefix_code* pc) {
  const int n = pc->n;
  int used = 0;
  int syms[2] = {0, 0};
  for (int i = 0; i < n; ++i) {
    if (counts[i]) {
      if (used < 2) syms[used] = i;
      if (++used > 2) break;
    }
  }

  if (used == 0) {
    /* Simple code with the single symbol 0. */
    wll_bw_put(bw, 0x01, 4);
    memset(pc->len, 0, (size_t)n);
    memset(pc->code, 0, (size_t)n * sizeof(pc->code[0]));
    return;
  }

  wll_huff_build(counts, n, WLL_MAX_CODE_LENGTH, pc->len);
  wll_huff_canonical(pc->len, n, pc->code);

  if (used <= 2 && syms[used - 1] < WLL_NUM_LITERAL_CODES) {
    wll_bw_put(bw, 1, 1); /* simple code */
    wll_bw_put(bw, (uint32_t)(used - 1), 1);
    if (syms[0] <= 1) {
      wll_bw_put(bw, 0, 1);
      wll_bw_put(bw, (uint32_t)syms[0], 1);
    } else {
      wll_bw_put(bw, 1, 1);
      wll_bw_put(bw, (uint32_t)syms[0], 8);
    }
    if (used == 2) wll_bw_put(bw, (uint32_t)syms[1], 8);
  } else {
    wll_bw_put(bw, 0, 1);
    store_full_code(bw, pc->len, n);
  }
  if (used == 1) pc->len[syms[0]] = 0;
}
