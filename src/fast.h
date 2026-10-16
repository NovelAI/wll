/* Fast encoding mode. */
#ifndef WLL_FAST_H
#define WLL_FAST_H

#include <stdint.h>

#include "bitwriter.h"
#include "mem.h"

/* Writes the transforms and the image data of the VP8L bitstream (everything
 * after the header) for RGBA pixels with rows `stride` bytes apart. Sets
 * *alpha_used if any alpha value is below 255. Returns 0 on allocation
 * failure. */
int wll_encode_fast(wll_bw* bw, const uint8_t* rgba, size_t stride, int width,
                    int height, wll_mem* mem, int* alpha_used);

#endif /* WLL_FAST_H */
