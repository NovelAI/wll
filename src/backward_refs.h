/* LZ77 backward references for VP8L. */
#ifndef WLL_BACKWARD_REFS_H
#define WLL_BACKWARD_REFS_H

#include <stdint.h>

#include "entropy.h"
#include "internal.h"

#define WLL_MAX_LENGTH_BITS 12
#define WLL_MAX_LENGTH ((1 << WLL_MAX_LENGTH_BITS) - 1)
#define WLL_WINDOW_SIZE ((1u << 20) - 120)

/* Best match found at every pixel: distance << WLL_MAX_LENGTH_BITS | length. */
typedef struct {
  uint32_t* offset_length;
  size_t size;
  wll_mem* mem;
} wll_hash_chain;

int wll_hash_chain_init(wll_hash_chain* hc, size_t size, wll_mem* mem);
void wll_hash_chain_free(wll_hash_chain* hc);

/* Finds a match for every pixel, trying up to `iter_max` candidates within
 * `window` pixels. Returns 0 on allocation failure. */
int wll_hash_chain_fill(wll_hash_chain* hc, const uint32_t* argb, int xsize,
                        int ysize, int iter_max, uint32_t window);

int wll_refs_init(wll_refs* r, size_t cap, wll_mem* mem);
void wll_refs_free(wll_refs* r);

/* Greedy LZ77 parse using the hash chain. `r` must have room for every pixel;
 * copy distances are pixel distances. */
void wll_refs_lz77(const uint32_t* argb, int xsize, int ysize,
                   const wll_hash_chain* hc, wll_refs* r);

/* Parse using only copies from the previous pixel and the pixel above (pixel
 * distances). `r` must have room for every pixel. */
void wll_refs_rle(const uint32_t* argb, int xsize, int ysize, wll_refs* r);

/* Re-parses the image as a shortest path over pixel positions, with symbol
 * costs taken from the statistics of `src` (pixel distances). Copies use the
 * hash chain's offset at their start. Returns 0 on allocation failure. */
int wll_refs_trace_backwards(const uint32_t* argb, int xsize, int ysize,
                             const wll_hash_chain* hc, const wll_refs* src,
                             wll_refs* dst, wll_mem* mem);

/* VP8L distance (plane) code of a pixel distance. */
uint32_t wll_distance_to_plane_code(int xsize, uint32_t dist);

/* Replaces pixel distances by VP8L distance (plane) codes. */
void wll_refs_to_plane_codes(wll_refs* r, int xsize);

/* Rough size in bits of `r` coded with one prefix code group. */
float wll_refs_estimate_bits(const wll_entropy_tab* t, const wll_refs* r,
                             int xsize);

#endif /* WLL_BACKWARD_REFS_H */
