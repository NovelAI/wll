/* Copyright 2026 Anlatan Inc.
 * SPDX-License-Identifier: Apache-2.0 */

/* VP8L transforms: subtract green, predictor, cross color. */
#ifndef WLL_TRANSFORMS_H
#define WLL_TRANSFORMS_H

#include <stddef.h>
#include <stdint.h>

#include "entropy.h"

#define WLL_NUM_PRED_MODES 14
#define WLL_ARGB_BLACK 0xff000000u

static inline int wll_subsample_size(int size, int bits) {
  return (size + (1 << bits) - 1) >> bits;
}

void wll_subtract_green(uint32_t* argb, size_t n);

/* Residuals of one row under predictor mode 11 (select), including the edge
 * rules: `up` is the row above, or NULL for the first row. */
#define WLL_PRED_SELECT 11
void wll_select_residual_row(const uint32_t* cur, const uint32_t* up,
                             int width, uint32_t* out);

/* Chooses a predictor for every (1 << bits)-sized tile of `argb` and writes
 * the residuals to `residuals`. `modes` receives one ARGB_BLACK | mode << 8
 * entry per tile. Returns 0 on allocation failure. */
int wll_predictor_transform(const uint32_t* argb, int width, int height,
                            int bits, uint32_t* modes, uint32_t* residuals);

/* Chooses cross color multipliers for every tile and applies them to `argb`
 * in place. `codes` receives one color code per tile. Returns 0 on
 * allocation failure. */
int wll_cross_color_transform(const wll_entropy_tab* t, uint32_t* argb,
                               int width, int height, int bits,
                               uint32_t* codes);

#endif /* WLL_TRANSFORMS_H */
