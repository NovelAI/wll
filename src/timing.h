/* Optional per-stage timing (build with -DWLL_TIMING). */
#ifndef WLL_TIMING_H
#define WLL_TIMING_H

enum {
  WLL_T_CONVERT,
  WLL_T_ANALYZE,
  WLL_T_PREDICTOR,
  WLL_T_CROSS_COLOR,
  WLL_T_HASH_CHAIN,
  WLL_T_PARSE,
  WLL_T_TRACE,
  WLL_T_CLUSTER,
  WLL_T_EMIT,
  /* Fast level. */
  WLL_T_F_RESIDUAL,
  WLL_T_F_MASKS,
  WLL_T_F_RLE,
  WLL_T_F_HISTO,
  WLL_T_F_EMIT,
  /* Decoder. */
  WLL_T_D_ENTROPY,
  WLL_T_D_ROWS,
  WLL_T_NUM
};

#ifdef WLL_TIMING
#include <time.h>

extern double wll_timing[WLL_T_NUM];
void wll_timing_report(void);

static inline double wll_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#define WLL_TIMER_START(name) double name = wll_now()
#define WLL_TIMER_LAP(name, stage)          \
  do {                                      \
    const double now_ = wll_now();          \
    wll_timing[stage] += now_ - (name);     \
    (name) = now_;                          \
  } while (0)
#else
#define WLL_TIMER_START(name) ((void)0)
#define WLL_TIMER_LAP(name, stage) ((void)0)
#endif

#endif /* WLL_TIMING_H */
