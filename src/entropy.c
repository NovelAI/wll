#include "entropy.h"

#include <pthread.h>
#if defined(__SSE2__)
#include <emmintrin.h>
#endif

static wll_entropy_tab g_tab;
static pthread_once_t g_tab_once = PTHREAD_ONCE_INIT;

static void tab_init(void) {
  g_tab.slog2[0] = 0.f;
  for (int i = 1; i < WLL_SLOG2_TABLE_SIZE; ++i) {
    g_tab.slog2[i] = (float)((double)i * log2((double)i));
  }
}

const wll_entropy_tab* wll_entropy_tab_get(void) {
  pthread_once(&g_tab_once, tab_init);
  return &g_tab;
}

void wll_acc_stats_init(const wll_entropy_tab* t, const uint32_t y[256],
                        wll_acc_stats* s) {
  double acc = 0.;
  uint32_t total = 0;
  for (int i = 0; i < 256; ++i) {
    acc += wll_slog2(t, y[i]);
    total += y[i];
  }
  s->slog_sum = acc;
  s->total = total;
}

double wll_combined_entropy(const wll_entropy_tab* t, const uint32_t x[256],
                            const uint32_t y[256], const wll_acc_stats* ys) {
  /* Bins where X is zero contribute their Y term, which ys->slog_sum holds
   * for all bins; correct it for the non-zero bins of X. */
  double acc = ys->slog_sum;
  uint32_t sum_x = 0;
  for (int i = 0; i < 256; i += 4) {
#if defined(__SSE2__)
    /* Tile histograms are sparse; skip groups of empty bins. */
    const __m128i v = _mm_loadu_si128((const __m128i*)(x + i));
    if (_mm_movemask_epi8(_mm_cmpeq_epi32(v, _mm_setzero_si128())) == 0xffff) {
      continue;
    }
#endif
    for (int k = i; k < i + 4; ++k) {
      const uint32_t xk = x[k];
      if (xk != 0) {
        sum_x += xk;
        acc += (double)wll_slog2(t, xk) + wll_slog2(t, xk + y[k]) -
               wll_slog2(t, y[k]);
      }
    }
  }
  return (double)wll_slog2(t, sum_x) + wll_slog2(t, sum_x + ys->total) - acc;
}

float wll_prediction_bias(const uint32_t counts[256], float weight_0,
                          float exp_val) {
  float bits = weight_0 * (float)counts[0];
  for (int i = 1; i < 16; ++i) {
    bits += exp_val * (float)(counts[i] + counts[256 - i]) / 100.f;
    exp_val *= 0.6f;
  }
  return -bits / 10.f;
}
