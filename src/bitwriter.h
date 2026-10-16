/* LSB-first bit writer, as used by the VP8L bitstream. */
#ifndef WLL_BITWRITER_H
#define WLL_BITWRITER_H

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "mem.h"

typedef struct {
  wll_mem* mem;
  uint8_t* buf;
  size_t pos;   /* bytes written to buf */
  size_t cap;
  uint64_t acc; /* pending bits, LSB first */
  int nbits;    /* number of pending bits, < 8 between calls */
  int error;    /* set when an allocation failed; output is then invalid */
} wll_bw;

int wll_bw_init(wll_bw* bw, size_t initial_cap, wll_mem* mem);
void wll_bw_free(wll_bw* bw);
/* Makes room for at least `extra` more bytes. Returns 0 on failure. */
int wll_bw_reserve(wll_bw* bw, size_t extra);
/* Pads the pending bits to a byte boundary and writes them out. */
void wll_bw_flush(wll_bw* bw);
/* Total number of bits written so far. */
static inline uint64_t wll_bw_bits(const wll_bw* bw) {
  return (uint64_t)bw->pos * 8 + (uint64_t)bw->nbits;
}

/* Appends the low `n` bits of `bits` (0 <= n <= 32). */
static inline void wll_bw_put(wll_bw* bw, uint32_t bits, int n) {
  assert(n >= 0 && n <= 32);
  assert(n == 32 || (bits >> n) == 0);
  /* Always store all 8 accumulator bytes, then keep the incomplete byte. */
  if (bw->cap - bw->pos < 8 && !wll_bw_reserve(bw, 8)) {
    bw->acc = 0;
    bw->nbits = 0;
    return;
  }
  uint64_t acc = bw->acc | (uint64_t)bits << bw->nbits;
  const int nbits = bw->nbits + n;
  uint8_t* p = bw->buf + bw->pos;
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  memcpy(p, &acc, 8);
#else
  for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(acc >> (8 * i));
#endif
  bw->pos += (size_t)(nbits >> 3);
  bw->acc = acc >> (nbits & ~7);
  bw->nbits = nbits & 7;
}

#endif /* WLL_BITWRITER_H */
