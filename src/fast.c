/* Fast encoding mode.
 *
 * Fixed choices instead of searches: subtract green, predictor mode 11
 * (select) everywhere, no cross color, copies only from the previous pixel
 * and the pixel above, and at most two prefix code groups: tiles whose alpha
 * residuals are all zero (where alpha then costs no bits) and the rest.
 *
 * Three passes over the image: residuals (fused with subtract green), then
 * finding copies while counting symbols, then writing the bits. Literals are
 * never stored as tokens; only the (rare) copies are listed.
 */
#include "fast.h"

#include <stdlib.h>
#include <string.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "backward_refs.h"
#include "huffman.h"
#include "internal.h"
#include "timing.h"
#include "transforms.h"

enum { PREDICTOR_TRANSFORM = 0, SUBTRACT_GREEN_TRANSFORM = 2 };
#define MIN_TRANSFORM_BITS 2
/* Largest predictor tiles: the mode image is constant anyway. */
#define PRED_BITS 9
/* Tile size of the prefix code groups. */
#define GROUP_BITS 5
#define GROUP_TILE (1 << GROUP_BITS)
/* Copy lengths. */
#define MIN_COPY 4
#define MAX_COPY 4095

/* Offsets of the five alphabets within a group's symbol arrays. */
enum {
  OFF_GREEN = 0,
  OFF_RED = OFF_GREEN + WLL_NUM_LITERAL_CODES + WLL_NUM_LENGTH_CODES,
  OFF_BLUE = OFF_RED + WLL_NUM_LITERAL_CODES,
  OFF_ALPHA = OFF_BLUE + WLL_NUM_LITERAL_CODES,
  OFF_DIST = OFF_ALPHA + WLL_NUM_LITERAL_CODES,
  GROUP_SYMBOLS = OFF_DIST + WLL_NUM_DISTANCE_CODES
};
static const int kAlphabetStart[6] = {OFF_GREEN, OFF_RED,  OFF_BLUE,
                                      OFF_ALPHA, OFF_DIST, GROUP_SYMBOLS};

typedef struct {
  uint32_t counts[GROUP_SYMBOLS];
  uint32_t packed[GROUP_SYMBOLS]; /* code | code length << 16 */
  /* Literal counts, added to `counts` after counting: green in four
   * interleaved copies, so that runs of equal literals do not make a chain
   * of increments of the same counter, and pairs of red and blue (red * 256
   * + blue, 65536 entries), which saves one increment per literal. */
  uint32_t green[4][256];
  uint32_t* red_blue;
  /* Whether red_blue then holds the joint red and blue codes (rb_codes). */
  int rb_coded;
} group;

/* Joint codes of red and blue: code | code length << RB_BITS, or RB_ESCAPE
 * if longer than RB_BITS. */
#define RB_BITS 27
#define RB_ESCAPE 0xffffffffu

WLL_TARGET_CLONES static void rb_codes(const uint32_t* pk, uint32_t* t) {
  for (int r = 0; r < 256; ++r) {
    const uint32_t rl = pk[OFF_RED + r] >> 16;
    const uint32_t rc = pk[OFF_RED + r] & 0xffff;
    uint32_t* row = t + (r << 8);
    for (int b = 0; b < 256; ++b) {
      const uint32_t l = rl + (pk[OFF_BLUE + b] >> 16);
      const uint32_t c = rc | (pk[OFF_BLUE + b] & 0xffff) << rl;
      row[b] = l > RB_BITS ? RB_ESCAPE : c | l << RB_BITS;
    }
  }
}

static inline void count_literal(uint32_t* green, uint32_t* red_blue,
                                 uint32_t v) {
  green[(v >> 8) & 0xff]++;
  red_blue[(v & 0xff) | ((v >> 8) & 0xff00)]++;
}

typedef struct {
  uint32_t pos;
  uint16_t len;
  uint8_t dist; /* 0: previous pixel, 1: pixel above */
} copy;

/* Writes the five prefix codes of `g` and fills its packed codes. */
static void group_store(wll_bw* bw, group* g) {
  uint8_t len[WLL_MAX_ALPHABET];
  uint16_t code[WLL_MAX_ALPHABET];
  for (int a = 0; a < 5; ++a) {
    const int start = kAlphabetStart[a];
    wll_prefix_code pc = {kAlphabetStart[a + 1] - start, len, code};
    wll_prefix_code_store(bw, g->counts + start, &pc);
    for (int s = 0; s < pc.n; ++s) {
      g->packed[start + s] = code[s] | (uint32_t)len[s] << 16;
    }
  }
}

/* ------------------------------------------------------------------------ */
/* Bit writing with the state in local variables. The caller makes sure 8
 * bytes past the current position are writable. */

typedef struct {
  uint8_t* p;
  uint64_t acc;
  int nbits; /* < 8 between calls */
} fast_bw;

/* Appends the low n bits of `bits` (n <= 56). */
static inline void fbw_put(fast_bw* w, uint64_t bits, int n) {
  w->acc |= bits << w->nbits;
  w->nbits += n;
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  memcpy(w->p, &w->acc, 8);
#else
  for (int i = 0; i < 8; ++i) w->p[i] = (uint8_t)(w->acc >> (8 * i));
#endif
  w->p += w->nbits >> 3;
  w->acc >>= w->nbits & ~7;
  w->nbits &= 7;
}

/* Appends n <= 64 bits. */
static inline void fbw_put_long(fast_bw* w, uint64_t bits, int n) {
  if (n <= 56) {
    fbw_put(w, bits, n);
  } else {
    fbw_put(w, bits & 0xffffffffu, 32);
    fbw_put(w, bits >> 32, n - 32);
  }
}

/* The four codes of a literal (at most 60 bits); `with_alpha` is 0 for
 * groups whose alpha code is empty. */
static inline uint64_t literal_bits(const uint32_t* pk, uint32_t v,
                                    int with_alpha, int* n) {
  const uint32_t g = pk[OFF_GREEN + ((v >> 8) & 0xff)];
  const uint32_t r = pk[OFF_RED + ((v >> 16) & 0xff)];
  const uint32_t b = pk[OFF_BLUE + (v & 0xff)];
  const int gl = (int)(g >> 16), gr = gl + (int)(r >> 16);
  const int grb = gr + (int)(b >> 16);
  uint64_t bits = (g & 0xffff) | (uint64_t)(r & 0xffff) << gl |
                  (uint64_t)(b & 0xffff) << gr;
  if (with_alpha) {
    const uint32_t a = pk[OFF_ALPHA + (v >> 24)];
    bits |= (uint64_t)(a & 0xffff) << grb;
    *n = grb + (int)(a >> 16);
  } else {
    *n = grb;
  }
  return bits;
}

/* Writes two literals with as few writes as possible. */
static inline void emit_pair(fast_bw* w, uint64_t b1, int n1, uint64_t b2,
                             int n2) {
  if (n1 + n2 <= 56) {
    fbw_put(w, b1 | b2 << n1, n1 + n2);
  } else {
    fbw_put_long(w, b1, n1);
    fbw_put_long(w, b2, n2);
  }
}

/* Writes n literals. Two literals usually fit one write, which halves the
 * chain of dependent bit writer updates. The writer state is copied to a
 * local so that it stays in registers: the byte stores could alias *out. */
static inline void emit_literals(fast_bw* out, const uint32_t* pk,
                                 const uint32_t* v, size_t n,
                                 int with_alpha) {
  fast_bw w = *out;
  size_t k = 0;
  for (; k + 2 <= n; k += 2) {
    int n1, n2;
    const uint64_t b1 = literal_bits(pk, v[k], with_alpha, &n1);
    const uint64_t b2 = literal_bits(pk, v[k + 1], with_alpha, &n2);
    emit_pair(&w, b1, n1, b2, n2);
  }
  if (k < n) {
    int n1;
    const uint64_t b1 = literal_bits(pk, v[k], with_alpha, &n1);
    fbw_put_long(&w, b1, n1);
  }
  *out = w;
}

/* literal_bits without alpha, with the joint red and blue codes `rb`. */
static inline uint64_t literal_bits_rb(const uint32_t* pk, const uint32_t* rb,
                                       uint32_t v, int* n) {
  const uint32_t g = pk[OFF_GREEN + ((v >> 8) & 0xff)];
  const uint32_t e = rb[(v & 0xff) | ((v >> 8) & 0xff00)];
  if (e == RB_ESCAPE) return literal_bits(pk, v, 0, n);
  const int gl = (int)(g >> 16);
  *n = gl + (int)(e >> RB_BITS);
  return (g & 0xffff) | (uint64_t)(e & ((1u << RB_BITS) - 1)) << gl;
}

/* emit_literals without alpha, with the joint red and blue codes `rb`. */
static inline void emit_literals_rb(fast_bw* out, const uint32_t* pk,
                                    const uint32_t* rb, const uint32_t* v,
                                    size_t n) {
  fast_bw w = *out;
  size_t k = 0;
  for (; k + 2 <= n; k += 2) {
    int n1, n2;
    const uint64_t b1 = literal_bits_rb(pk, rb, v[k], &n1);
    const uint64_t b2 = literal_bits_rb(pk, rb, v[k + 1], &n2);
    emit_pair(&w, b1, n1, b2, n2);
  }
  if (k < n) {
    int n1;
    const uint64_t b1 = literal_bits_rb(pk, rb, v[k], &n1);
    fbw_put_long(&w, b1, n1);
  }
  *out = w;
}

#if defined(__x86_64__) || defined(__i386__)
/* emit_literals, building the codes of 8 literals at a time. */
__attribute__((target("avx2,bmi2"))) static void emit_literals_avx2(
    fast_bw* out, const uint32_t* pk, const uint32_t* rb, const uint32_t* v,
    size_t n, int with_alpha) {
  fast_bw w = *out;
  const __m256i byte_hi = _mm256_set1_epi32(0xff00);
  const __m256i rb_mask = _mm256_set1_epi32((1 << RB_BITS) - 1);
  const __m256i byte = _mm256_set1_epi32(0xff);
  const __m256i low16 = _mm256_set1_epi32(0xffff);
  const __m256i low32 = _mm256_set1_epi64x(0xffffffff);
  const __m256i red = _mm256_set1_epi32(OFF_RED);
  const __m256i blue = _mm256_set1_epi32(OFF_BLUE);
  const __m256i alpha = _mm256_set1_epi32(OFF_ALPHA);
  const __m256i max32 = _mm256_set1_epi32(32);
  const __m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  for (size_t k = 0; k < n;) {
    const int count = n - k < 8 ? (int)(n - k) : 8;
    const __m256i px =
        count == 8 ? _mm256_loadu_si256((const __m256i*)(v + k))
                   : _mm256_maskload_epi32(
                         (const int*)(v + k),
                         _mm256_cmpgt_epi32(_mm256_set1_epi32(count), lanes));
    const __m256i gi = _mm256_and_si256(_mm256_srli_epi32(px, 8), byte);
    const __m256i g = _mm256_i32gather_epi32((const int*)pk, gi, 4);
    const __m256i gl = _mm256_srli_epi32(g, 16);
    __m256i len, code, slow;
    if (rb != NULL) {
      const __m256i rbi =
          _mm256_or_si256(_mm256_and_si256(px, byte),
                          _mm256_and_si256(_mm256_srli_epi32(px, 8), byte_hi));
      const __m256i e = _mm256_i32gather_epi32((const int*)rb, rbi, 4);
      len = _mm256_add_epi32(gl, _mm256_srli_epi32(e, RB_BITS));
      code = _mm256_or_si256(
          _mm256_and_si256(g, low16),
          _mm256_sllv_epi32(_mm256_and_si256(e, rb_mask), gl));
      slow = _mm256_cmpeq_epi32(e, _mm256_set1_epi32(-1));
    } else {
      const __m256i ri = _mm256_add_epi32(
          _mm256_and_si256(_mm256_srli_epi32(px, 16), byte), red);
      const __m256i bi = _mm256_add_epi32(_mm256_and_si256(px, byte), blue);
      const __m256i r = _mm256_i32gather_epi32((const int*)pk, ri, 4);
      const __m256i b = _mm256_i32gather_epi32((const int*)pk, bi, 4);
      const __m256i gr = _mm256_add_epi32(gl, _mm256_srli_epi32(r, 16));
      len = _mm256_add_epi32(gr, _mm256_srli_epi32(b, 16));
      code = _mm256_or_si256(
          _mm256_and_si256(g, low16),
          _mm256_or_si256(_mm256_sllv_epi32(_mm256_and_si256(r, low16), gl),
                          _mm256_sllv_epi32(_mm256_and_si256(b, low16), gr)));
      slow = _mm256_setzero_si256();
    }
    if (with_alpha) {
      const __m256i a = _mm256_i32gather_epi32(
          (const int*)pk, _mm256_add_epi32(_mm256_srli_epi32(px, 24), alpha),
          4);
      code = _mm256_or_si256(
          code, _mm256_sllv_epi32(_mm256_and_si256(a, low16), len));
      len = _mm256_add_epi32(len, _mm256_srli_epi32(a, 16));
    }
    slow = _mm256_or_si256(slow, _mm256_cmpgt_epi32(len, max32));
    if (_mm256_movemask_epi8(slow)) {
      /* Some literal needs more than 32 bits. */
      *out = w; /* keeps &w from escaping */
      if (rb != NULL) {
        emit_literals_rb(out, pk, rb, v + k, (size_t)count);
      } else {
        emit_literals(out, pk, v + k, (size_t)count, with_alpha);
      }
      w = *out;
      k += (size_t)count;
      continue;
    }
    /* Pairs of literals as 64-bit words. */
    const __m256i elen = _mm256_and_si256(len, low32);
    const __m256i pair = _mm256_or_si256(
        _mm256_and_si256(code, low32),
        _mm256_sllv_epi64(_mm256_srli_epi64(code, 32), elen));
    const __m256i plen = _mm256_add_epi64(elen, _mm256_srli_epi64(len, 32));
    uint64_t pairs[4], lens[4];
    _mm256_storeu_si256((__m256i*)pairs, pair);
    _mm256_storeu_si256((__m256i*)lens, plen);
    if (count == 8) {
      for (int p = 0; p < 4; ++p) fbw_put_long(&w, pairs[p], (int)lens[p]);
    } else {
      for (int p = 0; p < count >> 1; ++p) {
        fbw_put_long(&w, pairs[p], (int)lens[p]);
      }
      if (count & 1) {
        /* The last literal is the low half of its pair. */
        uint32_t codes[8], lens1[8];
        _mm256_storeu_si256((__m256i*)codes, code);
        _mm256_storeu_si256((__m256i*)lens1, len);
        fbw_put(&w, codes[count - 1], (int)lens1[count - 1]);
      }
    }
    k += (size_t)count;
  }
  *out = w;
}
#endif

/* ------------------------------------------------------------------------ */
/* Residuals */

static inline uint32_t sub_pixels(uint32_t a, uint32_t b) {
  uint32_t ag = 0x00ff00ffu + (a & 0xff00ff00u) - (b & 0xff00ff00u);
  uint32_t rb = 0xff00ff00u + (a & 0x00ff00ffu) - (b & 0x00ff00ffu);
  return (ag & 0xff00ff00u) | (rb & 0x00ff00ffu);
}

static inline int ch(uint32_t v, int shift) { return (int)((v >> shift) & 0xff); }

static inline uint32_t select_pred(uint32_t t, uint32_t l, uint32_t tl) {
  int pa_minus_pb = 0;
  for (int s = 0; s < 32; s += 8) {
    pa_minus_pb += abs(ch(l, s) - ch(tl, s)) - abs(ch(t, s) - ch(tl, s));
  }
  return pa_minus_pb <= 0 ? t : l;
}

/* Select residuals of row x in [x0, width), x0 >= 1, below another row. */
static void select_row_c(const uint32_t* cur, const uint32_t* up, int x0,
                         int width, uint32_t* out) {
  for (int x = x0; x < width; ++x) {
    out[x] = sub_pixels(cur[x], select_pred(up[x], cur[x - 1], up[x - 1]));
  }
}

#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2"))) static void select_row_avx2(
    const uint32_t* cur, const uint32_t* up, int width, uint32_t* out) {
  const __m256i ones8 = _mm256_set1_epi8(1);
  const __m256i ones16 = _mm256_set1_epi16(1);
  int x = 1;
  for (; x + 8 <= width; x += 8) {
    const __m256i l = _mm256_loadu_si256((const __m256i*)(cur + x - 1));
    const __m256i t = _mm256_loadu_si256((const __m256i*)(up + x));
    const __m256i tl = _mm256_loadu_si256((const __m256i*)(up + x - 1));
    const __m256i p = _mm256_loadu_si256((const __m256i*)(cur + x));
    /* Per-channel |L - TL| and |T - TL|, summed per pixel. */
    const __m256i dl =
        _mm256_or_si256(_mm256_subs_epu8(l, tl), _mm256_subs_epu8(tl, l));
    const __m256i dt =
        _mm256_or_si256(_mm256_subs_epu8(t, tl), _mm256_subs_epu8(tl, t));
    const __m256i sl =
        _mm256_madd_epi16(_mm256_maddubs_epi16(dl, ones8), ones16);
    const __m256i st =
        _mm256_madd_epi16(_mm256_maddubs_epi16(dt, ones8), ones16);
    /* Left wins only when its gradient sum is strictly larger. */
    const __m256i use_left = _mm256_cmpgt_epi32(sl, st);
    const __m256i pred = _mm256_blendv_epi8(t, l, use_left);
    _mm256_storeu_si256((__m256i*)(out + x), _mm256_sub_epi8(p, pred));
  }
  select_row_c(cur, up, x, width, out);
}
#endif

/* Converts a row of RGBA bytes to ARGB with green subtracted from red and
 * blue. ANDs the alpha values into the top byte of *all, which stays 0xff
 * while all pixels are opaque. */
static void convert_row_c(const uint8_t* src, int width, uint32_t* dst,
                          uint32_t* all) {
  uint32_t a = *all;
  for (int x = 0; x < width; ++x) {
    const uint8_t* p = src + 4 * x;
    const uint32_t g = p[1];
    dst[x] = (uint32_t)p[3] << 24 | ((p[0] - g) & 0xff) << 16 | g << 8 |
             ((p[2] - g) & 0xff);
    a &= (uint32_t)p[3] << 24 | 0x00ffffffu;
  }
  *all = a;
}

#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2"))) static void convert_row_avx2(
    const uint8_t* src, int width, uint32_t* dst, uint32_t* all) {
  /* RGBA bytes to B, G, R, A (little endian ARGB), and G to the R and B
   * positions. */
  const __m256i swap = _mm256_setr_epi8(
      2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15,
      2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15);
  const __m256i green = _mm256_setr_epi8(
      1, -1, 1, -1, 5, -1, 5, -1, 9, -1, 9, -1, 13, -1, 13, -1,
      1, -1, 1, -1, 5, -1, 5, -1, 9, -1, 9, -1, 13, -1, 13, -1);
  __m256i acc = _mm256_set1_epi32(-1);
  int x = 0;
  for (; x + 8 <= width; x += 8) {
    const __m256i v = _mm256_loadu_si256((const __m256i*)(src + 4 * x));
    acc = _mm256_and_si256(acc, v);
    const __m256i out = _mm256_sub_epi8(_mm256_shuffle_epi8(v, swap),
                                        _mm256_shuffle_epi8(v, green));
    _mm256_storeu_si256((__m256i*)(dst + x), out);
  }
  /* Only the alpha bytes (byte 3 of each pixel) matter. */
  acc = _mm256_or_si256(acc, _mm256_set1_epi32(0x00ffffff));
  const uint32_t simd_all =
      _mm256_testc_si256(acc, _mm256_set1_epi32(-1)) ? 0xffffffffu : 0;
  uint32_t tail = 0xffffffffu;
  convert_row_c(src + 4 * x, width - x, dst + x, &tail);
  *all &= simd_all & tail;
}
#endif

/* The select residuals of a row; `up` is NULL for the first row. */
static void residual_row(const uint32_t* cur, const uint32_t* up, int width,
                         uint32_t* out, int use_avx2) {
  if (up == NULL) {
    wll_select_residual_row(cur, NULL, width, out);
    return;
  }
  out[0] = sub_pixels(cur[0], up[0]);
#if defined(__x86_64__) || defined(__i386__)
  if (use_avx2) {
    select_row_avx2(cur, up, width, out);
    return;
  }
#endif
  (void)use_avx2;
  select_row_c(cur, up, 1, width, out);
}

/* ------------------------------------------------------------------------ */

/* Bit k of the masks (word k / 64, bit k % 64) tells whether pixel k equals
 * the previous pixel (eql) or the pixel above (equ), in reading order. These
 * set the bits of pixels [k0, k1). */
static void equal_masks_c(const uint32_t* res, size_t k0, size_t k1,
                          size_t width, uint64_t* eql, uint64_t* equ) {
  for (size_t k = k0; k < k1; ++k) {
    const uint64_t bit = (uint64_t)1 << (k & 63);
    if (k >= 1 && res[k] == res[k - 1]) eql[k >> 6] |= bit;
    if (k >= width && res[k] == res[k - width]) equ[k >> 6] |= bit;
  }
}

#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2"))) static void equal_masks_avx2(
    const uint32_t* res, size_t k0, size_t k1, size_t width, uint64_t* eql,
    uint64_t* equ) {
  /* The first row and unaligned ends take the scalar path. */
  size_t k = ((k0 > width ? k0 : width) + 7) & ~(size_t)7;
  if (k > k1) k = k1;
  equal_masks_c(res, k0, k, width, eql, equ);
  for (; k + 8 <= k1; k += 8) {
    const __m256i v = _mm256_loadu_si256((const __m256i*)(res + k));
    const __m256i l = _mm256_loadu_si256((const __m256i*)(res + k - 1));
    const __m256i u = _mm256_loadu_si256((const __m256i*)(res + k - width));
    const uint64_t ml = (uint64_t)_mm256_movemask_ps(
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(v, l)));
    const uint64_t mu = (uint64_t)_mm256_movemask_ps(
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(v, u)));
    eql[k >> 6] |= ml << (k & 63);
    equ[k >> 6] |= mu << (k & 63);
  }
  equal_masks_c(res, k, k1, width, eql, equ);
}
#endif

/* Turns the equality masks into copy candidates: bit k is set when four
 * pixels from k on all equal their left (or all their upper) neighbours,
 * i.e. when a copy of at least MIN_COPY pixels can start at k. */
static void copy_candidates(const uint64_t* eql, const uint64_t* equ,
                            size_t words, uint64_t* cand) {
  for (size_t w = 0; w < words; ++w) {
    const uint64_t l = eql[w], ln = eql[w + 1];
    const uint64_t u = equ[w], un = equ[w + 1];
    const uint64_t l4 = l & (l >> 1 | ln << 63) & (l >> 2 | ln << 62) &
                        (l >> 3 | ln << 61);
    const uint64_t u4 = u & (u >> 1 | un << 63) & (u >> 2 | un << 62) &
                        (u >> 3 | un << 61);
    cand[w] = l4 | u4;
  }
}

/* The number of consecutive set bits of `mask` from bit k on, at most
 * max_len (bits past the end of the image are clear). */
static inline int ones_from(const uint64_t* mask, size_t k, int max_len) {
  size_t w = k >> 6;
  const uint64_t inv = ~mask[w] >> (k & 63);
  int len;
  if (inv != 0) {
    len = __builtin_ctzll(inv);
  } else {
    len = 64 - (int)(k & 63);
    while (len < max_len) {
      const uint64_t m = ~mask[++w];
      if (m != 0) {
        len += __builtin_ctzll(m);
        break;
      }
      len += 64;
    }
  }
  return len < max_len ? len : max_len;
}

/* The first copy candidate in [k, end), or end. */
static inline size_t next_candidate(const uint64_t* cand, size_t k,
                                    size_t end) {
  size_t w = k >> 6;
  uint64_t bits = cand[w] & (~(uint64_t)0 << (k & 63));
  while (bits == 0) {
    if ((++w << 6) >= end) return end;
    bits = cand[w];
  }
  const size_t c = (w << 6) + (size_t)__builtin_ctzll(bits);
  return c < end ? c : end;
}

/* Pixels are processed in runs within one prefix code group: from pixel i
 * at column x, up to the next tile of the other group or the end of the row,
 * or with a single group, up to the end of the image. */
static inline size_t run_end(const uint8_t* tg_row, int tiles_x, int x,
                             int width, size_t i, size_t n, int num_groups) {
  if (num_groups == 1) return n;
  const int gi = tg_row[x >> GROUP_BITS];
  int t = (x >> GROUP_BITS) + 1;
  while (t < tiles_x && tg_row[t] == gi) ++t;
  const int x_end = t << GROUP_BITS < width ? t << GROUP_BITS : width;
  return i + (size_t)(x_end - x);
}

/* Moves column x (in a row of tile groups tg_row) on by `adv` pixels. */
static inline void advance(const uint8_t* tile_group, int tiles_x, int width,
                           size_t adv, int* x, int* y,
                           const uint8_t** tg_row) {
  *x += (int)(adv % (size_t)width);
  *y += (int)(adv / (size_t)width);
  if (*x >= width) {
    *x -= width;
    ++*y;
  }
  *tg_row = tile_group + (size_t)(*y >> GROUP_BITS) * (size_t)tiles_x;
}

/* Makes sure `bytes` bytes past the write position are writable. */
static int fbw_reserve(wll_bw* bw, fast_bw* w, size_t bytes) {
  if ((size_t)(bw->buf + bw->cap - w->p) >= bytes) return 1;
  bw->pos = (size_t)(w->p - bw->buf);
  if (!wll_bw_reserve(bw, bytes > (1 << 16) ? bytes : (1 << 16))) return 0;
  w->p = bw->buf + bw->pos;
  return 1;
}

/* Writes a prefix code whose only symbol is `sym` (< 256). */
static void put_single_symbol_code(wll_bw* bw, int sym) {
  wll_bw_put(bw, 1, 1); /* simple code */
  wll_bw_put(bw, 0, 1); /* one symbol */
  if (sym <= 1) {
    wll_bw_put(bw, 0, 1);
    wll_bw_put(bw, (uint32_t)sym, 1);
  } else {
    wll_bw_put(bw, 1, 1);
    wll_bw_put(bw, (uint32_t)sym, 8);
  }
}

/* Writes a sub-image whose pixels all equal 0x0000gg00 (only green set):
 * all five codes have a single symbol, so the pixels take no bits. */
static void put_constant_green_image(wll_bw* bw, int green) {
  wll_bw_put(bw, 0, 1); /* no color cache */
  put_single_symbol_code(bw, green);
  for (int i = 0; i < 4; ++i) put_single_symbol_code(bw, 0);
}

/* Writes a sub-image of group indices (0 or 1, in green), coded as literals
 * with one prefix code group. */
static void put_group_image(wll_bw* bw, group* g, const uint8_t* index,
                            size_t n) {
  memset(g->counts, 0, sizeof(g->counts));
  for (size_t i = 0; i < n; ++i) g->counts[OFF_GREEN + index[i]]++;
  g->counts[OFF_RED] = g->counts[OFF_BLUE] = g->counts[OFF_ALPHA] = 1;
  wll_bw_put(bw, 0, 1); /* no color cache */
  group_store(bw, g);
  for (size_t i = 0; i < n; ++i) {
    const uint32_t c = g->packed[OFF_GREEN + index[i]];
    wll_bw_put(bw, c & 0xffff, (int)(c >> 16));
  }
}

int wll_encode_fast(wll_bw* bw, const uint8_t* rgba, size_t stride, int width,
                    int height, wll_mem* mem, int* alpha_used) {
  const size_t n = (size_t)width * (size_t)height;
  const int tiles_x = wll_subsample_size(width, GROUP_BITS);
  const int tiles_y = wll_subsample_size(height, GROUP_BITS);
  const size_t tiles = (size_t)tiles_x * (size_t)tiles_y;
  int ok = 0;
  group* groups = (group*)calloc(2, sizeof(group));
  uint8_t* tile_group = (uint8_t*)calloc(tiles, 1);
  uint32_t* res = (uint32_t*)wll_mem_alloc(mem, n * sizeof(uint32_t));
  /* Two rows of converted pixels. */
  uint32_t* rows = (uint32_t*)malloc(2 * (size_t)width * sizeof(uint32_t));
  /* Copies are at least MIN_COPY pixels long. */
  copy* copies =
      (copy*)wll_mem_alloc(mem, (n / MIN_COPY + 1) * sizeof(copy));
  /* Equality masks and copy candidates, one bit per pixel plus padding. */
  uint64_t* masks = (uint64_t*)wll_mem_calloc(mem, 3 * (n / 64 + 2),
                                              sizeof(uint64_t));
  uint64_t* eql = masks;
  uint64_t* equ = masks ? masks + (n / 64 + 2) : NULL;
  uint64_t* cand = masks ? masks + 2 * (n / 64 + 2) : NULL;
  uint32_t* red_blue =
      (uint32_t*)wll_mem_calloc(mem, 2 * 65536, sizeof(uint32_t));
  if (groups == NULL || tile_group == NULL || res == NULL || rows == NULL ||
      copies == NULL || masks == NULL || red_blue == NULL) {
    goto done;
  }
  groups[0].red_blue = red_blue;
  groups[1].red_blue = red_blue + 65536;
#if (defined(__x86_64__) || defined(__i386__)) && !defined(WLL_DISABLE_AVX2)
  const int use_avx2 =
      __builtin_cpu_supports("avx2") && __builtin_cpu_supports("bmi2");
  /* Writing literals with gathers loses to scalar code where gathers are
   * slow. */
  const int use_gather = use_avx2 && !__builtin_cpu_is("znver1") &&
                         !__builtin_cpu_is("znver2");
#else
  const int use_avx2 = 0; /* WLL_DISABLE_AVX2 tests the portable code */
  const int use_gather = 0;
#endif

  WLL_TIMER_START(timer);
  /* Transforms. */
  wll_bw_put(bw, 1, 1);
  wll_bw_put(bw, SUBTRACT_GREEN_TRANSFORM, 2);
  wll_bw_put(bw, 1, 1);
  wll_bw_put(bw, PREDICTOR_TRANSFORM, 2);
  wll_bw_put(bw, PRED_BITS - MIN_TRANSFORM_BITS, 3);
  put_constant_green_image(bw, WLL_PRED_SELECT);
  wll_bw_put(bw, 0, 1); /* end of transforms */

  /* Pass 1: residuals and their equality masks; tiles with non-zero alpha
   * residuals go to group 1. */
  uint32_t alpha_all = 0xffffffffu;
  uint32_t* cur = rows;
  uint32_t* prev = rows + width;
  for (int y = 0; y < height; ++y) {
    const uint8_t* src = rgba + (size_t)y * stride;
#if defined(__x86_64__) || defined(__i386__)
    if (use_avx2) {
      convert_row_avx2(src, width, cur, &alpha_all);
    } else
#endif
    {
      convert_row_c(src, width, cur, &alpha_all);
    }
    uint32_t* out = res + (size_t)y * (size_t)width;
    residual_row(cur, y > 0 ? prev : NULL, width, out, use_avx2);
    const size_t k0 = (size_t)y * (size_t)width;
#if defined(__x86_64__) || defined(__i386__)
    if (use_avx2) {
      equal_masks_avx2(res, k0, k0 + (size_t)width, (size_t)width, eql, equ);
    } else
#endif
    {
      equal_masks_c(res, k0, k0 + (size_t)width, (size_t)width, eql, equ);
    }
    uint32_t* tmp = cur;
    cur = prev;
    prev = tmp;
    uint8_t* tg = tile_group + (size_t)(y >> GROUP_BITS) * (size_t)tiles_x;
    for (int x0 = 0; x0 < width; x0 += GROUP_TILE) {
      const int x1 = x0 + GROUP_TILE < width ? x0 + GROUP_TILE : width;
      uint32_t any = 0;
      for (int x = x0; x < x1; ++x) any |= out[x];
      if (any >> 24) tg[x0 >> GROUP_BITS] = 1;
    }
  }
  *alpha_used = (alpha_all >> 24) != 0xff;
  int num_groups = 1;
  /* Whether all alpha residuals of a group are zero. */
  int alpha_zero[2] = {1, 0};
  {
    size_t ones = 0;
    for (size_t i = 0; i < tiles; ++i) ones += tile_group[i];
    if (ones > 0 && ones < tiles) num_groups = 2;
    if (ones == tiles) alpha_zero[0] = 0;
    if (num_groups == 1) memset(tile_group, 0, tiles);
  }
  WLL_TIMER_LAP(timer, WLL_T_F_RESIDUAL);

  /* Distance symbols of the two copy distances. */
  int dist_sym[2], dist_nbits[2];
  uint32_t dist_extra[2];
  wll_prefix_encode(wll_distance_to_plane_code(width, 1), &dist_sym[0],
                    &dist_nbits[0], &dist_extra[0]);
  wll_prefix_encode(wll_distance_to_plane_code(width, (uint32_t)width),
                    &dist_sym[1], &dist_nbits[1], &dist_extra[1]);

  /* Pass 2: find copies (as libwebp's RLE parse) and count symbols. */
  copy_candidates(eql, equ, n / 64 + 1, cand);
  WLL_TIMER_LAP(timer, WLL_T_F_MASKS);
  size_t num_copies = 0;
  {
    size_t i = 0;
    int x = 0, y = 0;
    const uint8_t* tg_row = tile_group;
    while (i < n) {
      const size_t start = i;
      const size_t end =
          run_end(tg_row, tiles_x, x, width, i, n, num_groups);
      const int gi = tg_row[x >> GROUP_BITS];
      uint32_t* c = groups[gi].counts;
      uint32_t* const gr = groups[gi].green[0];
      uint32_t* const rb = groups[gi].red_blue;
      while (i < end) {
        /* Literals up to the next copy candidate. */
        const size_t lit_end = next_candidate(cand, i, end);
        size_t q = i;
        for (; q + 4 <= lit_end; q += 4) {
          count_literal(gr, rb, res[q]);
          count_literal(gr + 256, rb, res[q + 1]);
          count_literal(gr + 512, rb, res[q + 2]);
          count_literal(gr + 768, rb, res[q + 3]);
        }
        for (; q < lit_end; ++q) count_literal(gr, rb, res[q]);
        /* Known zero alpha residuals are counted in one go. */
        if (alpha_zero[gi]) {
          c[OFF_ALPHA] += (uint32_t)(lit_end - i);
        } else {
          for (q = i; q < lit_end; ++q) c[OFF_ALPHA + (res[q] >> 24)]++;
        }
        const size_t j = lit_end;
        if (j == end) {
          i = j;
          break;
        }
        /* As libwebp's RLE parse: the previous pixel unless the pixel
         * above gives a longer copy. */
        const int max_len = n - j < MAX_COPY ? (int)(n - j) : MAX_COPY;
        const int l1 = ones_from(eql, j, max_len);
        const int l2 = ones_from(equ, j, max_len);
        const int d = l1 >= l2 && l1 >= MIN_COPY ? 0 : 1;
        const int len = d ? l2 : l1;
        copies[num_copies].pos = (uint32_t)j;
        copies[num_copies].len = (uint16_t)len;
        copies[num_copies].dist = (uint8_t)d;
        ++num_copies;
        int code, nbits;
        uint32_t extra;
        wll_prefix_encode((uint32_t)len, &code, &nbits, &extra);
        c[OFF_GREEN + WLL_NUM_LITERAL_CODES + code]++;
        c[OFF_DIST + dist_sym[d]]++;
        i = j + (size_t)len;
      }
      if (num_groups > 1) {
        advance(tile_group, tiles_x, width, i - start, &x, &y, &tg_row);
      }
    }
  }
  for (int g = 0; g < num_groups; ++g) {
    uint32_t* c = groups[g].counts;
    for (int s = 0; s < 256; ++s) {
      for (int t = 0; t < 4; ++t) c[OFF_GREEN + s] += groups[g].green[t][s];
    }
    const uint32_t* rb = groups[g].red_blue;
    for (int r = 0; r < 256; ++r) {
      uint32_t sum = 0;
      for (int b = 0; b < 256; ++b) {
        sum += rb[r * 256 + b];
        c[OFF_BLUE + b] += rb[r * 256 + b];
      }
      c[OFF_RED + r] += sum;
    }
  }
  WLL_TIMER_LAP(timer, WLL_T_F_RLE);

  wll_bw_put(bw, 0, 1); /* no color cache */
  if (num_groups > 1) {
    group scratch;
    wll_bw_put(bw, 1, 1);
    wll_bw_put(bw, GROUP_BITS - 2, 3);
    put_group_image(bw, &scratch, tile_group, tiles);
  } else {
    wll_bw_put(bw, 0, 1);
  }
  for (int g = 0; g < num_groups; ++g) {
    group_store(bw, &groups[g]);
    /* Joint red and blue codes pay off for enough literals. */
    uint32_t literals = 0;
    for (int s = 0; s < 256; ++s) literals += groups[g].counts[OFF_GREEN + s];
    if (alpha_zero[g] && literals >= (1u << 16)) {
      rb_codes(groups[g].packed, groups[g].red_blue);
      groups[g].rb_coded = 1;    }
  }
  WLL_TIMER_LAP(timer, WLL_T_F_HISTO);

  /* Pass 3: write the pixels. */
  {
    /* Literals are written in chunks of at most CHUNK; each takes at most
     * 60 bits, plus the 8 bytes stored past the write position. */
    enum { CHUNK = 4096 };
    fast_bw w = {bw->buf + bw->pos, bw->acc, bw->nbits};
    size_t i = 0, ci = 0;
    int x = 0, y = 0;
    const uint8_t* tg_row = tile_group;
    while (i < n) {
      const size_t start = i;
      const size_t end =
          run_end(tg_row, tiles_x, x, width, i, n, num_groups);
      const int gi = tg_row[x >> GROUP_BITS];
      const uint32_t* pk = groups[gi].packed;
      while (i < end) {
        const size_t lit_end =
            ci < num_copies && copies[ci].pos < end ? copies[ci].pos : end;
        for (size_t q = i; q < lit_end;) {
          const size_t len = lit_end - q < CHUNK ? lit_end - q : CHUNK;
          if (!fbw_reserve(bw, &w, len * 8 + 16)) goto done;
#if defined(__x86_64__) || defined(__i386__)
          if (use_gather) {
            emit_literals_avx2(&w, pk,
                               groups[gi].rb_coded ? groups[gi].red_blue : NULL,
                               res + q, len, !alpha_zero[gi]);
          } else
#endif
          if (groups[gi].rb_coded) {
            emit_literals_rb(&w, pk, groups[gi].red_blue, res + q, len);
          } else {
            emit_literals(&w, pk, res + q, len, !alpha_zero[gi]);
          }
          q += len;
        }
        if (lit_end == end) {
          i = end;
          break;
        }
        const copy* cp = &copies[ci++];
        if (!fbw_reserve(bw, &w, 32)) goto done;
        int code, nbits;
        uint32_t extra;
        wll_prefix_encode(cp->len, &code, &nbits, &extra);
        const uint32_t lc = pk[OFF_GREEN + WLL_NUM_LITERAL_CODES + code];
        fbw_put(&w, (lc & 0xffff) | (uint64_t)extra << (lc >> 16),
                (int)(lc >> 16) + nbits);
        const uint32_t dc = pk[OFF_DIST + dist_sym[cp->dist]];
        fbw_put(&w,
                (dc & 0xffff) | (uint64_t)dist_extra[cp->dist] << (dc >> 16),
                (int)(dc >> 16) + dist_nbits[cp->dist]);
        i = lit_end + cp->len;
      }
      if (num_groups > 1) {
        advance(tile_group, tiles_x, width, i - start, &x, &y, &tg_row);
      }
    }
    bw->pos = (size_t)(w.p - bw->buf);
    bw->acc = w.acc;
    bw->nbits = w.nbits;
  }
  WLL_TIMER_LAP(timer, WLL_T_F_EMIT);
  ok = 1;
done:
  wll_mem_free(mem, red_blue);
  wll_mem_free(mem, masks);
  wll_mem_free(mem, copies);
  free(rows);
  wll_mem_free(mem, res);
  free(tile_group);
  free(groups);
  return ok;
}
