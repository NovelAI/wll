/* Copyright 2026 Anlatan Inc.
 * SPDX-License-Identifier: Apache-2.0 */

/* Entropy estimates used to steer encoder decisions. */
#ifndef WLL_ENTROPY_H
#define WLL_ENTROPY_H

#include <math.h>
#include <stdint.h>

#define WLL_SLOG2_TABLE_SIZE 65536

/* Table of n * log2(n) for small n. */
typedef struct {
  float slog2[WLL_SLOG2_TABLE_SIZE];
} wll_entropy_tab;

/* Returns the process-wide table, building it on first use. */
const wll_entropy_tab* wll_entropy_tab_get(void);

static inline float wll_slog2(const wll_entropy_tab* t, uint32_t n) {
  if (n < WLL_SLOG2_TABLE_SIZE) return t->slog2[n];
  return (float)((double)n * log2((double)n));
}

/* Summary of a histogram that many others get combined with. */
typedef struct {
  double slog_sum; /* sum of n * log2(n) over its bins */
  uint32_t total;
} wll_acc_stats;

void wll_acc_stats_init(const wll_entropy_tab* t, const uint32_t y[256],
                        wll_acc_stats* s);

/* Total entropy (bits) of X plus total entropy of X + Y, over 256 symbols,
 * where `ys` summarizes Y. Only visits the non-zero bins of X. */
double wll_combined_entropy(const wll_entropy_tab* t, const uint32_t x[256],
                            const uint32_t y[256], const wll_acc_stats* ys);

/* Negative cost favouring residuals near zero (libwebp's heuristic). */
float wll_prediction_bias(const uint32_t counts[256], float weight_0,
                          float exp_val);

#endif /* WLL_ENTROPY_H */
