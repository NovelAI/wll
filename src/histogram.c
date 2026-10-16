#include "histogram.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Clustering parameters, as libwebp uses them at quality 25. */
#define NUM_PARTITIONS 4
#define BIN_SIZE (NUM_PARTITIONS * NUM_PARTITIONS * NUM_PARTITIONS)
#define MIN_CLUSTER_SIZE 3
#define HISTO_QUEUE_SIZE 9

/* ------------------------------------------------------------------------ */
/* Histogram sets */

static size_t hist_words(int num_lit) {
  return (size_t)num_lit + 3 * 256 + WLL_NUM_DISTANCE_CODES;
}

int wll_hist_set_init(wll_hist_set* s, int max_size, int cache_bits,
                      wll_mem* mem) {
  s->num_lit = WLL_NUM_LITERAL_CODES + WLL_NUM_LENGTH_CODES +
               (cache_bits > 0 ? 1 << cache_bits : 0);
  s->size = 0;
  s->max_size = max_size;
  s->uf = NULL;
  s->mem = mem;
  const size_t words = hist_words(s->num_lit);
  s->data = (uint32_t*)wll_mem_calloc(mem, (size_t)max_size * words,
                                      sizeof(uint32_t));
  s->storage =
      (wll_hist*)wll_mem_calloc(mem, (size_t)max_size, sizeof(wll_hist));
  s->h = (wll_hist**)wll_mem_alloc(mem, (size_t)max_size * sizeof(wll_hist*));
  if (s->data == NULL || s->storage == NULL || s->h == NULL) {
    wll_hist_set_free(s);
    return 0;
  }
  for (int i = 0; i < max_size; ++i) {
    wll_hist* h = &s->storage[i];
    h->lit = s->data + (size_t)i * words;
    h->red = h->lit + s->num_lit;
    h->blue = h->red + 256;
    h->alpha = h->blue + 256;
    h->dist = h->alpha + 256;
    s->h[i] = h;
  }
  return 1;
}

void wll_hist_set_free(wll_hist_set* s) {
  wll_mem_free(s->mem, s->data);
  wll_mem_free(s->mem, s->storage);
  wll_mem_free(s->mem, s->h);
  s->data = NULL;
  s->storage = NULL;
  s->h = NULL;
  s->size = s->max_size = 0;
}

static void hist_clear(wll_hist* h, int num_lit) {
  memset(h->lit, 0, hist_words(num_lit) * sizeof(uint32_t));
}

static void hist_copy(const wll_hist* src, wll_hist* dst, int num_lit) {
  memcpy(dst->lit, src->lit, hist_words(num_lit) * sizeof(uint32_t));
  memcpy(dst->costs, src->costs, sizeof(src->costs));
  dst->bit_cost = src->bit_cost;
  memcpy(dst->trivial, src->trivial, sizeof(src->trivial));
  memcpy(dst->used, src->used, sizeof(src->used));
  dst->bin_id = src->bin_id;
}

static void hist_add_ref(wll_hist* h, const wll_ref* ref) {
  if (ref->kind == WLL_REF_LITERAL) {
    h->lit[(ref->v >> 8) & 0xff]++;
    h->red[(ref->v >> 16) & 0xff]++;
    h->blue[ref->v & 0xff]++;
    h->alpha[ref->v >> 24]++;
  } else if (ref->kind == WLL_REF_CACHE) {
    h->lit[WLL_NUM_LITERAL_CODES + WLL_NUM_LENGTH_CODES + ref->v]++;
  } else {
    int code, nbits;
    uint32_t extra;
    wll_prefix_encode(ref->len, &code, &nbits, &extra);
    h->lit[WLL_NUM_LITERAL_CODES + code]++;
    wll_prefix_encode(ref->v, &code, &nbits, &extra);
    h->dist[code]++;
  }
}

/* ------------------------------------------------------------------------ */
/* Cost model: refined entropy plus an estimate of the code description. */

typedef struct {
  double entropy;
  uint32_t sum, nonzeros, max_val;
  int nonzero_code;
} bit_entropy;

typedef struct {
  int counts[2];     /* number of streaks longer than 3, of zeros/non-zeros */
  int streaks[2][2]; /* total length of [zero/non-zero][short/long] streaks */
} streak_stats;

static inline void entropy_streak(const wll_entropy_tab* t, uint32_t val,
                                  int i, int i_prev, bit_entropy* be,
                                  streak_stats* st) {
  const int streak = i - i_prev;
  const int is_long = streak > 3;
  if (val != 0) {
    be->sum += val * (uint32_t)streak;
    be->nonzeros += (uint32_t)streak;
    be->nonzero_code = i_prev;
    be->entropy += (double)wll_slog2(t, val) * streak;
    if (be->max_val < val) be->max_val = val;
    st->counts[1] += is_long;
    st->streaks[1][is_long] += streak;
  } else {
    st->counts[0] += is_long;
    st->streaks[0][is_long] += streak;
  }
}

/* Entropy statistics of x (+ y if not NULL). */
static void get_entropy(const wll_entropy_tab* t, const uint32_t* x,
                        const uint32_t* y, int n, bit_entropy* be,
                        streak_stats* st) {
  memset(be, 0, sizeof(*be));
  memset(st, 0, sizeof(*st));
  int i_prev = 0;
  uint32_t prev = x[0] + (y ? y[0] : 0);
  int i;
  for (i = 1; i < n; ++i) {
    const uint32_t v = x[i] + (y ? y[i] : 0);
    if (v != prev) {
      entropy_streak(t, prev, i, i_prev, be, st);
      prev = v;
      i_prev = i;
    }
  }
  entropy_streak(t, prev, i, i_prev, be, st);
  be->entropy = (double)wll_slog2(t, be->sum) - be->entropy;
}

/* Shannon entropy pushed towards what a Huffman code can achieve. */
static double entropy_refine(const bit_entropy* be) {
  double mix;
  if (be->nonzeros < 5) {
    if (be->nonzeros <= 1) return 0.;
    if (be->nonzeros == 2) return 0.99 * be->sum + 0.01 * be->entropy;
    mix = be->nonzeros == 3 ? 0.95 : 0.7;
  } else {
    mix = 0.627;
  }
  double min_limit = 2. * be->sum - be->max_val;
  min_limit = mix * min_limit + (1. - mix) * be->entropy;
  return be->entropy < min_limit ? min_limit : be->entropy;
}

double wll_bits_entropy(const wll_entropy_tab* t, const uint32_t* counts,
                        int n) {
  bit_entropy be;
  streak_stats st;
  get_entropy(t, counts, NULL, n, &be, &st);
  return entropy_refine(&be);
}

static double huffman_header_cost(const streak_stats* st) {
  /* Code length code description, minus a bias for typical trimming. */
  double cost = 19 * 3 - 9.1;
  cost += (st->counts[0] * 1600 + 240 * st->streaks[0][1] +
           st->counts[1] * 2640 + 720 * st->streaks[1][1] +
           1840 * st->streaks[0][0] + 3360 * st->streaks[1][0]) /
          1024.;
  return cost;
}

static double population_cost(const wll_entropy_tab* t, const uint32_t* x,
                              int n, int* trivial, uint8_t* used) {
  bit_entropy be;
  streak_stats st;
  get_entropy(t, x, NULL, n, &be, &st);
  *trivial = be.nonzeros == 1 ? be.nonzero_code : -1;
  *used = st.streaks[1][0] != 0 || st.streaks[1][1] != 0;
  return entropy_refine(&be) + huffman_header_cost(&st);
}

static void compute_cost(const wll_entropy_tab* t, wll_hist* h, int num_lit) {
  h->bit_cost = 0.;
  for (int i = 0; i < 5; ++i) {
    int len;
    const uint32_t* p = wll_hist_population(h, i, num_lit, &len);
    h->costs[i] = population_cost(t, p, len, &h->trivial[i], &h->used[i]);
    h->bit_cost += h->costs[i];
  }
}

static double combined_entropy(const wll_entropy_tab* t, const wll_hist* a,
                               const wll_hist* b, int i, int num_lit) {
  const int trivial = a->trivial[i] >= 0 && a->trivial[i] == b->trivial[i];
  if (trivial || !a->used[i] || !b->used[i]) {
    return a->used[i] ? a->costs[i] : b->costs[i];
  }
  int len;
  const uint32_t* x = wll_hist_population(a, i, num_lit, &len);
  const uint32_t* y = wll_hist_population(b, i, num_lit, &len);
  bit_entropy be;
  streak_stats st;
  get_entropy(t, x, y, len, &be, &st);
  return entropy_refine(&be) + huffman_header_cost(&st);
}

/* Cost of a + b, abandoning the computation once it reaches `threshold`.
 * Returns 1 if the cost is below the threshold. */
static int combined_cost(const wll_entropy_tab* t, const wll_hist* a,
                         const wll_hist* b, double threshold, double* cost,
                         double costs[5], int num_lit) {
  if (threshold <= 0.) return 0;
  *cost = 0.;
  for (int i = 0; i < 5; ++i) {
    costs[i] = combined_entropy(t, a, b, i, num_lit);
    *cost += costs[i];
    if (*cost >= threshold) return 0;
  }
  return 1;
}

/* out = a + b; out may alias b. Costs of out are not updated. */
static void hist_add(const wll_hist* a, const wll_hist* b, wll_hist* out,
                     int num_lit) {
  for (int i = 0; i < 5; ++i) {
    int len;
    const uint32_t* pa = wll_hist_population(a, i, num_lit, &len);
    const uint32_t* pb = wll_hist_population(b, i, num_lit, &len);
    uint32_t* po = wll_hist_population(out, i, num_lit, &len);
    if (a->used[i] && b->used[i]) {
      for (int k = 0; k < len; ++k) po[k] = pa[k] + pb[k];
    } else if (a->used[i]) {
      memcpy(po, pa, (size_t)len * sizeof(*po));
    } else if (b->used[i]) {
      if (po != pb) memcpy(po, pb, (size_t)len * sizeof(*po));
    } else {
      memset(po, 0, (size_t)len * sizeof(*po));
    }
  }
  for (int i = 0; i < 5; ++i) {
    out->trivial[i] = a->trivial[i] == b->trivial[i] ? a->trivial[i] : -1;
    out->used[i] = a->used[i] || b->used[i];
  }
}

static void set_costs(wll_hist* h, double bit_cost, const double costs[5]) {
  h->bit_cost = bit_cost;
  memcpy(h->costs, costs, sizeof(h->costs));
}

/* out = a + b if C(a+b) - C(a) - C(b) < threshold. */
static int add_eval(const wll_entropy_tab* t, const wll_hist* a,
                    const wll_hist* b, wll_hist* out, double threshold,
                    int num_lit) {
  double cost, costs[5];
  threshold += a->bit_cost + b->bit_cost;
  if (!combined_cost(t, a, b, threshold, &cost, costs, num_lit)) return 0;
  hist_add(a, b, out, num_lit);
  set_costs(out, cost, costs);
  return 1;
}

static int uf_find(int* uf, int x) {
  while (uf[x] != x) {
    uf[x] = uf[uf[x]];
    x = uf[x];
  }
  return x;
}

/* Records that the tiles of `from` now belong to `into`. */
static void note_merge(wll_hist_set* s, const wll_hist* into,
                       const wll_hist* from) {
  if (s->uf) s->uf[uf_find(s->uf, from->label)] = uf_find(s->uf, into->label);
}

static void set_remove(wll_hist_set* s, int i) {
  s->h[i] = s->h[s->size - 1];
  --s->size;
}

/* ------------------------------------------------------------------------ */
/* Merging histograms with similar entropy */

static int bin_for(double min, double max, double val) {
  const double range = max - min;
  if (range <= 0.) return 0;
  int b = (int)((NUM_PARTITIONS - 1e-6) * (val - min) / range);
  return b < 0 ? 0 : b >= NUM_PARTITIONS ? NUM_PARTITIONS - 1 : b;
}

static void analyze_entropy_bins(wll_hist_set* s) {
  double lo[3] = {INFINITY, INFINITY, INFINITY};
  double hi[3] = {0., 0., 0.};
  static const int kIdx[3] = {WLL_H_LITERAL, WLL_H_RED, WLL_H_BLUE};
  for (int i = 0; i < s->size; ++i) {
    for (int k = 0; k < 3; ++k) {
      const double c = s->h[i]->costs[kIdx[k]];
      if (c < lo[k]) lo[k] = c;
      if (c > hi[k]) hi[k] = c;
    }
  }
  for (int i = 0; i < s->size; ++i) {
    int bin = 0;
    for (int k = 0; k < 3; ++k) {
      bin = bin * NUM_PARTITIONS + bin_for(lo[k], hi[k], s->h[i]->costs[kIdx[k]]);
    }
    s->h[i]->bin_id = bin;
  }
}

static void combine_entropy_bins(const wll_entropy_tab* t, wll_hist_set* s,
                                 wll_hist** scratch, double cost_factor) {
  struct {
    int first;
    int failures;
  } bins[BIN_SIZE];
  const int max_failures = 32;
  for (int b = 0; b < BIN_SIZE; ++b) {
    bins[b].first = -1;
    bins[b].failures = 0;
  }
  for (int idx = 0; idx < s->size;) {
    const int bin = s->h[idx]->bin_id;
    const int first = bins[bin].first;
    if (first == -1) {
      bins[bin].first = idx++;
      continue;
    }
    const double thresh = -s->h[idx]->bit_cost * cost_factor;
    if (add_eval(t, s->h[first], s->h[idx], *scratch, thresh, s->num_lit)) {
      /* Merge only if the result stays trivial in red, blue and alpha, or
       * both inputs were already non-trivial somewhere there. */
      const wll_hist* c = *scratch;
      const wll_hist* a = s->h[idx];
      const wll_hist* f = s->h[first];
      int try_combine = c->trivial[WLL_H_RED] >= 0 &&
                        c->trivial[WLL_H_BLUE] >= 0 &&
                        c->trivial[WLL_H_ALPHA] >= 0;
      if (!try_combine) {
        try_combine = (a->trivial[WLL_H_RED] < 0 || a->trivial[WLL_H_BLUE] < 0 ||
                       a->trivial[WLL_H_ALPHA] < 0) &&
                      (f->trivial[WLL_H_RED] < 0 || f->trivial[WLL_H_BLUE] < 0 ||
                       f->trivial[WLL_H_ALPHA] < 0);
      }
      if (try_combine || bins[bin].failures >= max_failures) {
        wll_hist* tmp = *scratch;
        tmp->label = s->h[first]->label;
        note_merge(s, tmp, s->h[idx]);
        *scratch = s->h[first];
        s->h[first] = tmp;
        set_remove(s, idx);
      } else {
        ++bins[bin].failures;
        ++idx;
      }
    } else {
      ++idx;
    }
  }
}

/* ------------------------------------------------------------------------ */
/* Priority queue of histogram pairs; queue[0] is the best pair. */

typedef struct {
  int idx1, idx2;
  double cost_diff;
  double cost_combo;
  double costs[5];
} hist_pair;

typedef struct {
  hist_pair* q;
  int size, max_size;
} pair_queue;

static int queue_init(pair_queue* pq, int max_size) {
  pq->size = 0;
  pq->max_size = max_size;
  pq->q = (hist_pair*)malloc((size_t)(max_size + 1) * sizeof(hist_pair));
  return pq->q != NULL;
}

static void queue_pop(pair_queue* pq, hist_pair* p) {
  *p = pq->q[pq->size - 1];
  --pq->size;
}

static void queue_update_head(pair_queue* pq, hist_pair* p) {
  if (p->cost_diff < pq->q[0].cost_diff) {
    hist_pair tmp = pq->q[0];
    pq->q[0] = *p;
    *p = tmp;
  }
}

static void pair_fix(int bad, int good, hist_pair* p) {
  if (p->idx1 == bad) p->idx1 = good;
  if (p->idx2 == bad) p->idx2 = good;
  if (p->idx1 > p->idx2) {
    int tmp = p->idx1;
    p->idx1 = p->idx2;
    p->idx2 = tmp;
  }
}

static int pair_update(const wll_entropy_tab* t, const wll_hist* h1,
                       const wll_hist* h2, double threshold, hist_pair* p,
                       int num_lit) {
  const double sum = h1->bit_cost + h2->bit_cost;
  if (!combined_cost(t, h1, h2, threshold + sum, &p->cost_combo, p->costs,
                     num_lit)) {
    return 0;
  }
  p->cost_diff = p->cost_combo - sum;
  return 1;
}

/* Adds the pair if merging it gains more than -threshold. Returns its cost
 * difference, or 0 if not added. */
static double queue_push(const wll_entropy_tab* t, pair_queue* pq,
                         wll_hist_set* s, int idx1, int idx2,
                         double threshold) {
  if (pq->size == pq->max_size) return 0.;
  if (idx1 > idx2) {
    int tmp = idx1;
    idx1 = idx2;
    idx2 = tmp;
  }
  hist_pair p;
  p.idx1 = idx1;
  p.idx2 = idx2;
  if (!pair_update(t, s->h[idx1], s->h[idx2], threshold, &p, s->num_lit)) {
    return 0.;
  }
  pq->q[pq->size++] = p;
  queue_update_head(pq, &pq->q[pq->size - 1]);
  return p.cost_diff;
}

static void merge_pair(wll_hist_set* s, const hist_pair* p) {
  note_merge(s, s->h[p->idx1], s->h[p->idx2]);
  hist_add(s->h[p->idx2], s->h[p->idx1], s->h[p->idx1], s->num_lit);
  set_costs(s->h[p->idx1], p->cost_combo, p->costs);
  set_remove(s, p->idx2);
}

static uint32_t lehmer_rand(uint32_t* seed) {
  *seed = (uint32_t)(((uint64_t)*seed * 48271u) % 2147483647u);
  return *seed;
}

static int combine_stochastic(const wll_entropy_tab* t, wll_hist_set* s,
                              int* do_greedy) {
  if (s->size < MIN_CLUSTER_SIZE) {
    *do_greedy = 1;
    return 1;
  }
  pair_queue pq;
  if (!queue_init(&pq, HISTO_QUEUE_SIZE)) return 0;
  uint32_t seed = 1;
  int tries_without_success = 0;
  const int outer_iters = s->size;
  const int max_tries_without_success = outer_iters / 2;
  for (int iter = 0; iter < outer_iters && s->size >= MIN_CLUSTER_SIZE &&
                     ++tries_without_success < max_tries_without_success;
       ++iter) {
    double best_cost = pq.size == 0 ? 0. : pq.q[0].cost_diff;
    const uint32_t rand_range = (uint32_t)(s->size - 1) * (uint32_t)s->size;
    /* libwebp tries size / 2 pairs; fewer cost little compression. */
    const int num_tries = s->size / 8;
    for (int j = 0; s->size >= 2 && j < num_tries; ++j) {
      const uint32_t r = lehmer_rand(&seed) % rand_range;
      uint32_t idx1 = r / (uint32_t)(s->size - 1);
      uint32_t idx2 = r % (uint32_t)(s->size - 1);
      if (idx2 >= idx1) ++idx2;
      const double c = queue_push(t, &pq, s, (int)idx1, (int)idx2, best_cost);
      if (c < 0.) {
        best_cost = c;
        if (pq.size == pq.max_size) break;
      }
    }
    if (pq.size == 0) continue;

    const hist_pair best = pq.q[0];
    merge_pair(s, &best);
    for (int j = 0; j < pq.size;) {
      hist_pair* p = &pq.q[j];
      const int a_best = p->idx1 == best.idx1 || p->idx1 == best.idx2;
      const int b_best = p->idx2 == best.idx1 || p->idx2 == best.idx2;
      if (a_best && b_best) {
        queue_pop(&pq, p);
        continue;
      }
      /* The merged pair is now best.idx1, and the histogram that was last
       * (index s->size) moved to best.idx2. */
      pair_fix(best.idx2, best.idx1, p);
      pair_fix(s->size, best.idx2, p);
      if ((a_best || b_best) &&
          !pair_update(t, s->h[p->idx1], s->h[p->idx2], 0., p, s->num_lit)) {
        queue_pop(&pq, p);
        continue;
      }
      queue_update_head(&pq, p);
      ++j;
    }
    tries_without_success = 0;
  }
  *do_greedy = s->size <= MIN_CLUSTER_SIZE;
  free(pq.q);
  return 1;
}

static int combine_greedy(const wll_entropy_tab* t, wll_hist_set* s) {
  pair_queue pq;
  const int n = s->size;
  if (!queue_init(&pq, n * n)) return 0;
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) queue_push(t, &pq, s, i, j, 0.);
  }
  while (pq.size > 0) {
    const hist_pair best = pq.q[0];
    merge_pair(s, &best);
    for (int i = 0; i < pq.size;) {
      hist_pair* p = &pq.q[i];
      if (p->idx1 == best.idx1 || p->idx2 == best.idx1 ||
          p->idx1 == best.idx2 || p->idx2 == best.idx2) {
        queue_pop(&pq, p);
      } else {
        pair_fix(s->size, best.idx2, p);
        queue_update_head(&pq, p);
        ++i;
      }
    }
    for (int i = 0; i < s->size; ++i) {
      if (i != best.idx1) queue_push(t, &pq, s, best.idx1, i, 0.);
    }
  }
  free(pq.q);
  return 1;
}

/* Entropy summary of one population of a cluster. */
typedef struct {
  double slog_sum; /* sum of n * log2(n) */
  uint32_t sum, nonzeros, max_val;
  double refined;  /* entropy_refine() of the above */
} pop_stats;

static void pop_stats_init(const wll_entropy_tab* t, const uint32_t* v,
                           int len, pop_stats* s) {
  memset(s, 0, sizeof(*s));
  for (int j = 0; j < len; ++j) {
    if (v[j]) {
      s->slog_sum += wll_slog2(t, v[j]);
      s->sum += v[j];
      s->nonzeros++;
      if (v[j] > s->max_val) s->max_val = v[j];
    }
  }
  bit_entropy be = {wll_slog2(t, s->sum) - s->slog_sum, s->sum, s->nonzeros,
                    s->max_val, 0};
  s->refined = entropy_refine(&be);
}

/* Returns the cluster whose cost grows least when the tile histogram `in`
 * is added to it. The growth is the change of the refined entropy, updated
 * incrementally from the tile's non-zero bins; the code description estimate
 * is taken as unchanged. Evaluation starts with `first`, the cluster the tile
 * was merged into, which is usually best and lets the other candidates stop
 * early. */
static int remap_pick(const wll_entropy_tab* t, const wll_hist* in,
                      const wll_hist_set* out, int first,
                      const pop_stats* cstats) {
  uint16_t idx[5][WLL_MAX_ALPHABET];
  int nnz[5];
  uint32_t sum_in[5];
  for (int p = 0; p < 5; ++p) {
    int len;
    const uint32_t* v = wll_hist_population(in, p, out->num_lit, &len);
    nnz[p] = 0;
    sum_in[p] = 0;
    for (int j = 0; j < len; ++j) {
      if (v[j]) {
        idx[p][nnz[p]++] = (uint16_t)j;
        sum_in[p] += v[j];
      }
    }
  }
  int best_out = first;
  double best = INFINITY;
  for (int jj = -1; jj < out->size; ++jj) {
    const int k = jj < 0 ? first : jj;
    if (jj == first || k < 0) continue;
    const wll_hist* o = out->h[k];
    double delta = 0.;
    int p;
    for (p = 0; p < 5; ++p) {
      if (!in->used[p]) continue;
      if (!o->used[p]) {
        delta += in->costs[p] - o->costs[p];
      } else if (!(in->trivial[p] >= 0 && in->trivial[p] == o->trivial[p])) {
        int len;
        const uint32_t* ov = wll_hist_population(o, p, out->num_lit, &len);
        const uint32_t* iv = wll_hist_population(in, p, out->num_lit, &len);
        const pop_stats* s = &cstats[k * 5 + p];
        double sl = s->slog_sum;
        uint32_t nz = s->nonzeros, mx = s->max_val;
        for (int q = 0; q < nnz[p]; ++q) {
          const int j = idx[p][q];
          const uint32_t a = ov[j], b = a + iv[j];
          sl += (double)wll_slog2(t, b) - wll_slog2(t, a);
          nz += a == 0;
          if (b > mx) mx = b;
        }
        const uint32_t sum = s->sum + sum_in[p];
        bit_entropy be = {wll_slog2(t, sum) - sl, sum, nz, mx, 0};
        delta += entropy_refine(&be) - s->refined;
      }
      if (delta >= best) break;
    }
    if (p == 5) {
      best = delta;
      best_out = k;
    }
  }
  return best_out;
}

/* Assigns every tile histogram to the cluster it adds the least cost to, then
 * rebuilds the clusters from their tiles. Empty tiles (NULL) copy the
 * previous tile's cluster. `home` maps tile labels to cluster indices. */
static int remap(const wll_entropy_tab* t, wll_hist* const* in, int in_size,
                 wll_hist_set* out, uint32_t* symbols, int* home) {
  if (out->size > 1) {
    pop_stats* cstats =
        (pop_stats*)malloc((size_t)out->size * 5 * sizeof(pop_stats));
    if (cstats == NULL) return 0;
    for (int i = 0; i < in_size; ++i) home[i] = -1;
    for (int k = 0; k < out->size; ++k) {
      home[uf_find(out->uf, out->h[k]->label)] = k;
    }
    for (int k = 0; k < out->size; ++k) {
      for (int p = 0; p < 5; ++p) {
        int len;
        const uint32_t* v =
            wll_hist_population(out->h[k], p, out->num_lit, &len);
        pop_stats_init(t, v, len, &cstats[k * 5 + p]);
      }
    }
    for (int i = 0; i < in_size; ++i) {
      if (in[i] == NULL) {
        symbols[i] = i > 0 ? symbols[i - 1] : 0;
        continue;
      }
      symbols[i] = (uint32_t)remap_pick(t, in[i], out,
                                        home[uf_find(out->uf, i)], cstats);
    }
    free(cstats);
  } else {
    for (int i = 0; i < in_size; ++i) symbols[i] = 0;
  }
  for (int k = 0; k < out->size; ++k) {
    hist_clear(out->h[k], out->num_lit);
    memset(out->h[k]->used, 0, sizeof(out->h[k]->used));
    for (int i = 0; i < 5; ++i) out->h[k]->trivial[i] = -1;
  }
  for (int i = 0; i < in_size; ++i) {
    if (in[i] == NULL) continue;
    wll_hist* o = out->h[symbols[i]];
    for (int k = 0; k < 5; ++k) {
      int len;
      const uint32_t* src = wll_hist_population(in[i], k, out->num_lit, &len);
      uint32_t* dst = wll_hist_population(o, k, out->num_lit, &len);
      for (int j = 0; j < len; ++j) dst[j] += src[j];
    }
  }
  return 1;
}

static double combine_cost_factor(int histo_size) {
  double f = 16.;
  if (histo_size > 256) f /= 2;
  if (histo_size > 512) f /= 2;
  if (histo_size > 1024) f /= 2;
  f /= 2; /* quality <= 50 */
  return f / 100.;
}

int wll_histogram_image(const wll_entropy_tab* t, const wll_refs* refs,
                        int xsize, int ysize, int histo_bits, int cache_bits,
                        wll_hist_set* out, uint32_t* symbols, wll_mem* mem) {
  const int tiles_x = (xsize + (1 << histo_bits) - 1) >> histo_bits;
  const int tiles_y = (ysize + (1 << histo_bits) - 1) >> histo_bits;
  const int tiles = tiles_x * tiles_y;
  wll_hist_set orig;
  wll_hist** in = NULL;
  int* uf = NULL;
  int ok = 0;
  /* One spare histogram serves as scratch space for merges. */
  if (!wll_hist_set_init(out, tiles + 1, cache_bits, mem)) return 0;
  if (!wll_hist_set_init(&orig, tiles, cache_bits, mem)) goto done;
  in = (wll_hist**)malloc((size_t)tiles * sizeof(*in));
  /* Union-find parents, followed by scratch space for remap(). */
  uf = (int*)malloc(2 * (size_t)tiles * sizeof(*uf));
  if (in == NULL || uf == NULL) goto done;
  for (int i = 0; i < tiles; ++i) uf[i] = i;
  out->uf = uf;
  const int num_lit = orig.num_lit;

  /* Per tile histograms; each reference counts in the tile it starts in. */
  {
    int x = 0, y = 0;
    for (size_t i = 0; i < refs->size; ++i) {
      const wll_ref* ref = &refs->refs[i];
      hist_add_ref(orig.h[(y >> histo_bits) * tiles_x + (x >> histo_bits)], ref);
      x += ref->len;
      while (x >= xsize) {
        x -= xsize;
        ++y;
      }
    }
  }
  wll_hist* scratch = out->h[tiles];
  out->size = 0;
  for (int i = 0; i < tiles; ++i) {
    wll_hist* h = orig.h[i];
    compute_cost(t, h, num_lit);
    if (!h->used[0] && !h->used[1] && !h->used[2] && !h->used[3] &&
        !h->used[4]) {
      in[i] = NULL;
    } else {
      in[i] = h;
      wll_hist* c = out->h[out->size++];
      hist_copy(h, c, num_lit);
      c->label = i;
    }
  }

  if (out->size > 2 * BIN_SIZE) {
    analyze_entropy_bins(out);
    combine_entropy_bins(t, out, &scratch, combine_cost_factor(tiles));
  }
  int do_greedy;
  if (!combine_stochastic(t, out, &do_greedy)) goto done;
  if (do_greedy && !combine_greedy(t, out)) goto done;
  if (!remap(t, in, tiles, out, symbols, uf + tiles)) goto done;
  ok = 1;
done:
  out->uf = NULL;
  free(uf);
  free(in);
  wll_hist_set_free(&orig);
  if (!ok) wll_hist_set_free(out);
  return ok;
}
