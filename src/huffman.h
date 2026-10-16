/* Prefix (Huffman) codes for the VP8L bitstream. */
#ifndef WLL_HUFFMAN_H
#define WLL_HUFFMAN_H

#include <stdint.h>

#include "bitwriter.h"

#define WLL_MAX_CODE_LENGTH 15
#define WLL_NUM_LITERAL_CODES 256
#define WLL_NUM_LENGTH_CODES 24
#define WLL_NUM_DISTANCE_CODES 40
#define WLL_MAX_CACHE_BITS 11
#define WLL_MAX_ALPHABET \
  (WLL_NUM_LITERAL_CODES + WLL_NUM_LENGTH_CODES + (1 << WLL_MAX_CACHE_BITS))

typedef struct {
  int n;          /* alphabet size */
  uint8_t* len;   /* bits emitted per symbol (0 for single-symbol codes) */
  uint16_t* code; /* bit-reversed canonical code per symbol */
} wll_prefix_code;

/* Computes Huffman code lengths of at most `max_len` bits for `counts`.
 * Unused symbols get length 0. A lone used symbol gets length 1. */
void wll_huff_build(const uint32_t* counts, int n, int max_len, uint8_t* len);

/* Assigns bit-reversed canonical codes for `len`. */
void wll_huff_canonical(const uint8_t* len, int n, uint16_t* code);

/* Builds a code for `counts`, writes its description to `bw` and fills `pc`
 * (whose arrays must hold pc->n entries) for emitting symbols. */
void wll_prefix_code_store(wll_bw* bw, const uint32_t* counts,
                           wll_prefix_code* pc);

static inline void wll_put_symbol(wll_bw* bw, const wll_prefix_code* pc,
                                  int sym) {
  wll_bw_put(bw, pc->code[sym], pc->len[sym]);
}

#endif /* WLL_HUFFMAN_H */
