#include "backward_refs.h"

#include <stdlib.h>
#include <string.h>
#if defined(__SSE2__)
#include <emmintrin.h>
#endif

#define HASH_BITS 18
#define HASH_SIZE (1 << HASH_BITS)
/* Shortest copy that beats coding the pixels as literals. */
#define MIN_LENGTH 4

/* Distance to plane code mapping for the 2D neighbourhood (from libwebp). */
static const uint8_t kPlaneToCodeLut[128] = {
    96,  73,  55,  39,  23, 13, 5,  1,  255, 255, 255, 255, 255, 255, 255, 255,
    101, 78,  58,  42,  26, 16, 8,  2,  0,   3,   9,   17,  27,  43,  59,  79,
    102, 86,  62,  46,  32, 20, 10, 6,  4,   7,   11,  21,  33,  47,  63,  87,
    105, 90,  70,  52,  37, 28, 18, 14, 12,  15,  19,  29,  38,  53,  71,  91,
    110, 99,  82,  66,  48, 35, 30, 24, 22,  25,  31,  36,  49,  67,  83,  100,
    115, 108, 94,  76,  64, 50, 44, 40, 34,  41,  45,  51,  65,  77,  95,  109,
    118, 113, 103, 92,  80, 68, 60, 56, 54,  57,  61,  69,  81,  93,  104, 114,
    119, 116, 111, 106, 97, 88, 84, 74, 72,  75,  85,  89,  98,  107, 112, 117};

uint32_t wll_distance_to_plane_code(int xsize, uint32_t dist) {
  const int yoffset = (int)(dist / (uint32_t)xsize);
  const int xoffset = (int)(dist - (uint32_t)yoffset * (uint32_t)xsize);
  if (xoffset <= 8 && yoffset < 8) {
    return kPlaneToCodeLut[yoffset * 16 + 8 - xoffset] + 1u;
  } else if (xoffset > xsize - 8 && yoffset < 7) {
    return kPlaneToCodeLut[(yoffset + 1) * 16 + 8 + (xsize - xoffset)] + 1u;
  }
  return dist + 120;
}

/* Number of equal leading pixels of a and b, at most max_len. */
static inline int match_length(const uint32_t* a, const uint32_t* b,
                               int max_len) {
  int i = 0;
#if defined(__SSE2__)
  for (; i + 4 <= max_len; i += 4) {
    const __m128i eq = _mm_cmpeq_epi32(_mm_loadu_si128((const __m128i*)(a + i)),
                                       _mm_loadu_si128((const __m128i*)(b + i)));
    const int mask = _mm_movemask_epi8(eq);
    if (mask != 0xffff) return i + (__builtin_ctz((unsigned)~mask) >> 2);
  }
#endif
  while (i < max_len && a[i] == b[i]) ++i;
  return i;
}

/* Like match_length, but returns 0 early when position `best` differs, since
 * the match then cannot beat a match of length `best`. */
static inline int match_length_above(const uint32_t* a, const uint32_t* b,
                                     int best, int max_len) {
  if (a[best] != b[best]) return 0;
  return match_length(a, b, max_len);
}

static inline uint32_t pix_pair_hash(uint32_t a0, uint32_t a1) {
  uint32_t key = a1 * 0xc6a4a793u + a0 * 0x5bd1e996u;
  return key >> (32 - HASH_BITS);
}

int wll_hash_chain_init(wll_hash_chain* hc, size_t size, wll_mem* mem) {
  hc->size = size;
  hc->mem = mem;
  hc->offset_length = (uint32_t*)wll_mem_alloc(mem, size * sizeof(uint32_t));
  return hc->offset_length != NULL;
}

void wll_hash_chain_free(wll_hash_chain* hc) {
  wll_mem_free(hc->mem, hc->offset_length);
  hc->offset_length = NULL;
  hc->size = 0;
}

int wll_hash_chain_fill(wll_hash_chain* hc, const uint32_t* argb, int xsize,
                        int ysize, int iter_max, uint32_t window) {
  const int size = xsize * ysize;
  uint32_t* ol = hc->offset_length;
  if (size <= 2) {
    ol[0] = ol[size - 1] = 0;
    return 1;
  }
  int32_t* head = (int32_t*)wll_mem_alloc(hc->mem, HASH_SIZE * sizeof(int32_t));
  /* Chain of earlier positions with the same hash; reuses the output array. */
  int32_t* chain = (int32_t*)ol;
  if (head == NULL) return 0;
  memset(head, 0xff, HASH_SIZE * sizeof(int32_t));

  int pos = 0;
  int same = argb[0] == argb[1];
  while (pos < size - 2) {
    const int same_next = argb[pos + 1] == argb[pos + 2];
    if (same && same_next) {
      /* Runs of one color would all share a hash; hash color and remaining
       * run length instead. */
      uint32_t len = 1;
      while (pos + (int)len + 2 < size && argb[pos + len + 2] == argb[pos]) {
        ++len;
      }
      if (len > WLL_MAX_LENGTH) {
        /* Those pixels are matched at distance 1 by the search below. */
        for (uint32_t k = 0; k < len - WLL_MAX_LENGTH; ++k) chain[pos + k] = -1;
        pos += (int)(len - WLL_MAX_LENGTH);
        len = WLL_MAX_LENGTH;
      }
      while (len) {
        const uint32_t h = pix_pair_hash(argb[pos], len--);
        chain[pos] = head[h];
        head[h] = pos++;
      }
      same = 0;
    } else {
      const uint32_t h = pix_pair_hash(argb[pos], argb[pos + 1]);
      chain[pos] = head[h];
      head[h] = pos++;
      same = same_next;
    }
  }
  chain[pos] = head[pix_pair_hash(argb[pos], argb[pos + 1])];
  wll_mem_free(hc->mem, head);

  /* Best match per position, walking backwards so that matches can be
   * extended to the left cheaply. The first pixel has nothing to its left and
   * the last nothing to match after it. */
  ol[0] = ol[size - 1] = 0;
  for (int base = size - 2; base > 0;) {
    const int remaining = size - 1 - base;
    const int max_len = remaining < WLL_MAX_LENGTH ? remaining : WLL_MAX_LENGTH;
    const uint32_t* start = argb + base;
    const int min_pos = (uint32_t)base > window ? base - (int)window : 0;
    const int length_max = max_len < 256 ? max_len : 256;
    int iter = iter_max;
    int best_length = 0;
    uint32_t best_distance = 0;

    int p = chain[base];
    /* Try the pixel above and the previous pixel first. */
    if (base >= xsize) {
      const int l = match_length_above(start - xsize, start, best_length, max_len);
      if (l > best_length) {
        best_length = l;
        best_distance = (uint32_t)xsize;
      }
      --iter;
    }
    {
      const int l = match_length_above(start - 1, start, best_length, max_len);
      if (l > best_length) {
        best_length = l;
        best_distance = 1;
      }
      --iter;
    }
    if (best_length == WLL_MAX_LENGTH) p = min_pos - 1;
    uint32_t best_argb = start[best_length];

    for (; p >= min_pos && --iter; p = chain[p]) {
      if (argb[p + best_length] != best_argb) continue;
      const int l = match_length(argb + p, start, max_len);
      if (l > best_length) {
        best_length = l;
        best_distance = (uint32_t)(base - p);
        best_argb = start[best_length];
        if (best_length >= length_max) break;
      }
    }

    /* While the match also extends to the left, it is the best match of the
     * preceding positions too. */
    int max_base = base;
    for (;;) {
      ol[base] = best_distance << WLL_MAX_LENGTH_BITS | (uint32_t)best_length;
      --base;
      if (best_distance == 0 || base == 0) break;
      if ((uint32_t)base < best_distance ||
          argb[base - (int)best_distance] != argb[base]) {
        break;
      }
      /* At the length limit a closer match may exist; keep looking unless
       * the distance is already minimal. */
      if (best_length == WLL_MAX_LENGTH && best_distance != 1 &&
          base + WLL_MAX_LENGTH < max_base) {
        break;
      }
      if (best_length < WLL_MAX_LENGTH) {
        ++best_length;
        max_base = base;
      }
    }
  }
  return 1;
}

int wll_refs_init(wll_refs* r, size_t cap, wll_mem* mem) {
  r->size = 0;
  r->cap = cap;
  r->mem = mem;
  r->refs = (wll_ref*)wll_mem_alloc(mem, cap * sizeof(*r->refs));
  return r->refs != NULL;
}

void wll_refs_free(wll_refs* r) {
  wll_mem_free(r->mem, r->refs);
  r->refs = NULL;
  r->size = r->cap = 0;
}

static inline void add_literal(wll_refs* r, uint32_t argb) {
  r->refs[r->size++] = (wll_ref){argb, 1, WLL_REF_LITERAL};
}

static inline void add_copy(wll_refs* r, uint32_t dist, int len) {
  r->refs[r->size++] = (wll_ref){dist, (uint16_t)len, WLL_REF_COPY};
}

static inline int hc_length(const wll_hash_chain* hc, int pos) {
  return (int)(hc->offset_length[pos] & ((1u << WLL_MAX_LENGTH_BITS) - 1));
}

static inline uint32_t hc_offset(const wll_hash_chain* hc, int pos) {
  return hc->offset_length[pos] >> WLL_MAX_LENGTH_BITS;
}

void wll_refs_lz77(const uint32_t* argb, int xsize, int ysize,
                   const wll_hash_chain* hc, wll_refs* r) {
  const int n = xsize * ysize;
  int last_check = -1;
  r->size = 0;
  for (int i = 0; i < n;) {
    int len = hc_length(hc, i);
    const uint32_t offset = hc_offset(hc, i);
    if (len >= MIN_LENGTH) {
      /* Shorten the copy if that lets the next match reach further. */
      const int j_max = i + len >= n ? n - 1 : i + len;
      int max_reach = 0;
      last_check = i > last_check ? i : last_check;
      for (int j = last_check + 1; j <= j_max; ++j) {
        const int len_j = hc_length(hc, j);
        const int reach = j + (len_j >= MIN_LENGTH ? len_j : 1);
        if (reach > max_reach) {
          len = j - i;
          max_reach = reach;
          if (max_reach >= n) break;
        }
      }
    } else {
      len = 1;
    }
    if (len == 1) {
      add_literal(r, argb[i]);
    } else {
      add_copy(r, offset, len);
    }
    i += len;
  }
}


void wll_refs_rle(const uint32_t* argb, int xsize, int ysize, wll_refs* r) {
  const int n = xsize * ysize;
  r->size = 0;
  add_literal(r, argb[0]);
  for (int i = 1; i < n;) {
    const int max_len = n - i < WLL_MAX_LENGTH ? n - i : WLL_MAX_LENGTH;
    const int rle_len = match_length(argb + i, argb + i - 1, max_len);
    const int row_len =
        i < xsize ? 0 : match_length(argb + i, argb + i - xsize, max_len);
    if (rle_len >= row_len && rle_len >= MIN_LENGTH) {
      add_copy(r, 1, rle_len);
      i += rle_len;
    } else if (row_len >= MIN_LENGTH) {
      add_copy(r, (uint32_t)xsize, row_len);
      i += row_len;
    } else {
      add_literal(r, argb[i]);
      ++i;
    }
  }
}

/* ------------------------------------------------------------------------ */
/* Cost-based parse */

/* Lengths relaxed per pushed copy; long runs are re-pushed halfway. */
#define TRACE_MAX_PUSH 512

typedef struct {
  float lit[WLL_NUM_LITERAL_CODES + WLL_NUM_LENGTH_CODES];
  float red[256], blue[256], alpha[256];
  float dist[WLL_NUM_DISTANCE_CODES];
} cost_model;

/* Bits per symbol as -log2 of its frequency; all zero for trivial codes. */
static void bit_estimates(const uint32_t* counts, int n, float* out) {
  uint32_t sum = 0;
  int nonzeros = 0;
  for (int i = 0; i < n; ++i) {
    sum += counts[i];
    nonzeros += counts[i] != 0;
  }
  if (nonzeros <= 1) {
    for (int i = 0; i < n; ++i) out[i] = 0.f;
    return;
  }
  const float log_sum = log2f((float)sum);
  for (int i = 0; i < n; ++i) {
    out[i] = counts[i] ? log_sum - log2f((float)counts[i]) : log_sum;
  }
}

static int cost_model_build(cost_model* m, const wll_refs* r, int xsize) {
  wll_histo* h = (wll_histo*)calloc(1, sizeof(*h));
  if (h == NULL) return 0;
  for (size_t i = 0; i < r->size; ++i) {
    const wll_ref* ref = &r->refs[i];
    int code, nbits;
    uint32_t ev;
    if (ref->kind == WLL_REF_LITERAL) {
      h->green[(ref->v >> 8) & 0xff]++;
      h->red[(ref->v >> 16) & 0xff]++;
      h->blue[ref->v & 0xff]++;
      h->alpha[ref->v >> 24]++;
    } else {
      wll_prefix_encode(ref->len, &code, &nbits, &ev);
      h->green[WLL_NUM_LITERAL_CODES + code]++;
      wll_prefix_encode(wll_distance_to_plane_code(xsize, ref->v), &code, &nbits,
                        &ev);
      h->dist[code]++;
    }
  }
  bit_estimates(h->green, WLL_NUM_LITERAL_CODES + WLL_NUM_LENGTH_CODES, m->lit);
  bit_estimates(h->red, 256, m->red);
  bit_estimates(h->blue, 256, m->blue);
  bit_estimates(h->alpha, 256, m->alpha);
  bit_estimates(h->dist, WLL_NUM_DISTANCE_CODES, m->dist);
  free(h);
  return 1;
}

static inline float literal_cost(const cost_model* m, uint32_t v) {
  return m->alpha[v >> 24] + m->red[(v >> 16) & 0xff] +
         m->lit[(v >> 8) & 0xff] + m->blue[v & 0xff];
}

int wll_refs_trace_backwards(const uint32_t* argb, int xsize, int ysize,
                             const wll_hash_chain* hc, const wll_refs* src,
                             wll_refs* dst, wll_mem* mem) {
  const int n = xsize * ysize;
  cost_model* m = (cost_model*)malloc(sizeof(*m));
  /* Path costs grow to the size of the whole image in bits; float would
   * lose the resolution to tell close candidates apart. */
  double* cost = (double*)wll_mem_alloc(mem, (size_t)n * sizeof(*cost));
  uint16_t* step = (uint16_t*)wll_mem_alloc(mem, (size_t)n * sizeof(*step));
  float* len_cost = (float*)malloc((WLL_MAX_LENGTH + 1) * sizeof(*len_cost));
  int ok = 0;
  if (m == NULL || cost == NULL || step == NULL || len_cost == NULL) goto done;
  if (!cost_model_build(m, src, xsize)) goto done;
  for (int k = 1; k <= WLL_MAX_LENGTH; ++k) {
    int code, nbits;
    uint32_t ev;
    wll_prefix_encode((uint32_t)k, &code, &nbits, &ev);
    len_cost[k] = m->lit[WLL_NUM_LITERAL_CODES + code] + (float)nbits;
  }
  for (int i = 0; i < n; ++i) cost[i] = INFINITY;

  /* cost[i]: cheapest coding of pixels [0, i]; step[i]: length of the last
   * element (1 = literal). */
  uint32_t prev_offset = 0;
  int next_push = 0;
  for (int i = 0; i < n; ++i) {
    const double prev_cost = i > 0 ? cost[i - 1] : 0.;
    /* Literal costs are discounted, as libwebp does. */
    const double c = prev_cost + 0.82f * literal_cost(m, argb[i]);
    if (c < cost[i]) {
      cost[i] = c;
      step[i] = 1;
    }
    const int len = hc_length(hc, i);
    const uint32_t offset = hc_offset(hc, i);
    if (len >= 2 && (offset != prev_offset || i >= next_push)) {
      int code, nbits;
      uint32_t ev;
      wll_prefix_encode(wll_distance_to_plane_code(xsize, offset), &code, &nbits,
                        &ev);
      const double base = prev_cost + m->dist[code] + (float)nbits;
      const int kmax = len < TRACE_MAX_PUSH ? len : TRACE_MAX_PUSH;
      for (int k = 2; k <= kmax; ++k) {
        const double ck = base + len_cost[k];
        if (ck < cost[i + k - 1]) {
          cost[i + k - 1] = ck;
          step[i + k - 1] = (uint16_t)k;
        }
      }
      /* Later pixels with the same offset continue this match; only push
       * again when it was cut short. */
      next_push = len > TRACE_MAX_PUSH ? i + TRACE_MAX_PUSH / 2 : i + len;
    }
    prev_offset = len >= 2 ? offset : 0;
  }

  /* Walk the chosen path back, filling dst from its end, then move the
   * references to the front. There are at most n of them. */
  size_t j = (size_t)n;
  for (int i = n - 1; i >= 0;) {
    const int len = step[i];
    const int start = i - len + 1;
    dst->refs[--j] = len == 1 ? (wll_ref){argb[i], 1, WLL_REF_LITERAL}
                              : (wll_ref){hc_offset(hc, start), (uint16_t)len,
                                          WLL_REF_COPY};
    i = start - 1;
  }
  dst->size = (size_t)n - j;
  memmove(dst->refs, dst->refs + j, dst->size * sizeof(*dst->refs));
  ok = 1;
done:
  free(len_cost);
  wll_mem_free(mem, step);
  wll_mem_free(mem, cost);
  free(m);
  return ok;
}

void wll_refs_to_plane_codes(wll_refs* r, int xsize) {
  for (size_t i = 0; i < r->size; ++i) {
    if (r->refs[i].kind == WLL_REF_COPY) {
      r->refs[i].v = wll_distance_to_plane_code(xsize, r->refs[i].v);
    }
  }
}

static float population_bits(const wll_entropy_tab* t, const uint32_t* h,
                             int n) {
  uint32_t sum = 0;
  float acc = 0.f;
  int used = 0;
  for (int i = 0; i < n; ++i) {
    if (h[i]) {
      sum += h[i];
      acc += wll_slog2(t, h[i]);
      ++used;
    }
  }
  if (used <= 1) return 0.f;
  /* Shannon entropy plus a rough charge for describing the code. */
  return wll_slog2(t, sum) - acc + 4.f * (float)used;
}

float wll_refs_estimate_bits(const wll_entropy_tab* t, const wll_refs* r,
                             int xsize) {
  wll_histo* h = (wll_histo*)calloc(1, sizeof(*h));
  if (h == NULL) return 0.f;
  float extra = 0.f;
  for (size_t i = 0; i < r->size; ++i) {
    const wll_ref* ref = &r->refs[i];
    if (ref->kind == WLL_REF_LITERAL) {
      h->green[(ref->v >> 8) & 0xff]++;
      h->red[(ref->v >> 16) & 0xff]++;
      h->blue[ref->v & 0xff]++;
      h->alpha[ref->v >> 24]++;
    } else {
      int code, nbits;
      uint32_t ev;
      wll_prefix_encode(ref->len, &code, &nbits, &ev);
      h->green[WLL_NUM_LITERAL_CODES + code]++;
      extra += (float)nbits;
      wll_prefix_encode(wll_distance_to_plane_code(xsize, ref->v), &code, &nbits,
                        &ev);
      h->dist[code]++;
      extra += (float)nbits;
    }
  }
  float bits = extra +
               population_bits(t, h->green, WLL_NUM_LITERAL_CODES +
                                                WLL_NUM_LENGTH_CODES) +
               population_bits(t, h->red, WLL_NUM_LITERAL_CODES) +
               population_bits(t, h->blue, WLL_NUM_LITERAL_CODES) +
               population_bits(t, h->alpha, WLL_NUM_LITERAL_CODES) +
               population_bits(t, h->dist, WLL_NUM_DISTANCE_CODES);
  free(h);
  return bits;
}
