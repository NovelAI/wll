/* Symbol histograms and their clustering into prefix code groups. */
#ifndef WLL_HISTOGRAM_H
#define WLL_HISTOGRAM_H

#include <stdint.h>

#include "entropy.h"
#include "internal.h"

enum { WLL_H_LITERAL, WLL_H_RED, WLL_H_BLUE, WLL_H_ALPHA, WLL_H_DIST };

typedef struct {
  uint32_t* lit; /* green, length prefixes and cache indices */
  uint32_t* red;
  uint32_t* blue;
  uint32_t* alpha;
  uint32_t* dist;
  double costs[5];
  double bit_cost;
  int trivial[5]; /* the only used symbol, or -1 */
  uint8_t used[5];
  int bin_id;
  int label; /* a tile merged into this histogram (see wll_hist_set.uf) */
} wll_hist;

typedef struct {
  int num_lit;   /* literal alphabet size */
  int size;      /* histograms in use */
  int max_size;
  wll_hist** h;  /* h[0..size) are in use */
  wll_hist* storage;
  uint32_t* data;
  int* uf; /* union-find parents over tile labels, or NULL */
  wll_mem* mem;
} wll_hist_set;

int wll_hist_set_init(wll_hist_set* s, int max_size, int cache_bits,
                      wll_mem* mem);
void wll_hist_set_free(wll_hist_set* s);

static inline uint32_t* wll_hist_population(const wll_hist* h, int i,
                                            int num_lit, int* len) {
  switch (i) {
    case WLL_H_LITERAL: *len = num_lit; return h->lit;
    case WLL_H_RED: *len = 256; return h->red;
    case WLL_H_BLUE: *len = 256; return h->blue;
    case WLL_H_ALPHA: *len = 256; return h->alpha;
    default: *len = WLL_NUM_DISTANCE_CODES; return h->dist;
  }
}

/* Entropy of `counts` adjusted towards what a Huffman code can reach. */
double wll_bits_entropy(const wll_entropy_tab* t, const uint32_t* counts,
                        int n);

/* Clusters the per-tile histograms of `refs` (tiles of 1 << histo_bits) into
 * prefix code groups. On return `out` holds one histogram per group and
 * symbols[tile] the group of each tile. `out` must have room for every tile.
 * Returns 0 on allocation failure. */
int wll_histogram_image(const wll_entropy_tab* t, const wll_refs* refs,
                        int xsize, int ysize, int histo_bits, int cache_bits,
                        wll_hist_set* out, uint32_t* symbols, wll_mem* mem);

#endif /* WLL_HISTOGRAM_H */
