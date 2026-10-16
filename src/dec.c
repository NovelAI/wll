/* Copyright 2026 Anlatan Inc.
 * Parts derived from libwebp, Copyright (c) 2010, Google Inc. (see NOTICE).
 * SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause */

/* Lossless WebP (VP8L) decoder: all lossless features (the four
 * transforms, meta prefix codes, LZ77 copies and color caches), in simple
 * or extended (VP8X) files. Lossy and animated files are rejected as
 * unsupported.
 *
 * The input is untrusted: every code is validated before its lookup table is
 * built, table sizes are computed before filling, and all copies and reads
 * are bounds checked. Reading past the end of the data is an error.
 */
#include <stdlib.h>
#include <string.h>
#if defined(__SSE2__)
#include <emmintrin.h>
#endif

#include "internal.h"
#include "mem.h"
#include "timing.h"
#include "wll.h"

#define NUM_LITERALS 256
#define NUM_LENGTH_CODES 24
#define NUM_DISTANCE_CODES 40
#define GREEN_ALPHABET (NUM_LITERALS + NUM_LENGTH_CODES)
#define MAX_CACHE_BITS 11
#define MAX_GREEN_ALPHABET (GREEN_ALPHABET + (1 << MAX_CACHE_BITS))
#define MAX_CODE_LENGTH 15
#define NUM_CODE_LENGTH_CODES 19
#define ARGB_BLACK 0xff000000u

enum { GREEN = 0, RED, BLUE, ALPHA, DIST, CODES_PER_GROUP };
enum {
  PREDICTOR_TRANSFORM = 0,
  CROSS_COLOR_TRANSFORM = 1,
  SUBTRACT_GREEN_TRANSFORM = 2,
  COLOR_INDEXING_TRANSFORM = 3
};

static const int kAlphabetSize[CODES_PER_GROUP] = {
    GREEN_ALPHABET, NUM_LITERALS, NUM_LITERALS, NUM_LITERALS,
    NUM_DISTANCE_CODES};

/* ------------------------------------------------------------------------ */
/* Bit reader (LSB first) */

/* buf and pos are not adjacent: GCC 13 otherwise packs them into one vector
 * register around the stores before calls, which puts vector moves on the
 * decoding loop's critical path. */
typedef struct {
  const uint8_t* data;
  size_t size;
  uint64_t buf;  /* bits [0, nbits) are valid */
  int nbits;
  size_t pos;    /* next byte to load; may pass size (zeros are loaded) */
} bitreader;

static inline uint64_t load64le(const uint8_t* p) {
  uint64_t v;
  memcpy(&v, p, 8);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  v = __builtin_bswap64(v);
#endif
  return v;
}

/* Makes at least 56 bits available. */
static inline void br_refill(bitreader* br) {
  if (br->pos + 8 <= br->size) {
    br->buf |= load64le(br->data + br->pos) << br->nbits;
    br->pos += (size_t)((63 - br->nbits) >> 3);
    br->nbits |= 56;
  } else {
    while (br->nbits <= 56) {
      const uint64_t byte = br->pos < br->size ? br->data[br->pos] : 0;
      br->buf |= byte << br->nbits;
      ++br->pos;
      br->nbits += 8;
    }
  }
}

/* The next n <= 32 bits; the caller makes sure they are available. */
static inline uint32_t br_peek(const bitreader* br, int n) {
  return (uint32_t)(br->buf & (((uint64_t)1 << n) - 1));
}

static inline void br_skip(bitreader* br, int n) {
  br->buf >>= n;
  br->nbits -= n;
}

static inline uint32_t br_take(bitreader* br, int n) {
  const uint32_t v = br_peek(br, n);
  br_skip(br, n);
  return v;
}

/* Reads n <= 32 bits, refilling as needed (for headers). */
static uint32_t br_read(bitreader* br, int n) {
  if (br->nbits < n) br_refill(br);
  return br_take(br, n);
}

/* Whether more bits were consumed than the data holds. */
static inline int br_overrun(const bitreader* br) {
  return (uint64_t)br->pos * 8 - (uint64_t)br->nbits >
         (uint64_t)br->size * 8;
}

/* ------------------------------------------------------------------------ */
/* Prefix code lookup tables
 *
 * A table starts with 1 << root_bits entries indexed by the next root_bits
 * bits. An entry either gives a symbol and the number of bits to consume, or
 * (sub != 0) points at a second-level table of 1 << sub entries, indexed by
 * the bits after the root bits, at offset `value` from the table start. */

typedef struct {
  uint16_t value;
  uint8_t bits;
  uint8_t sub;
} hentry;

typedef struct {
  uint32_t offset; /* into the decoder's entry array */
  uint8_t root_bits;
} hcode;

typedef struct {
  hentry* entries;
  size_t size, cap;
} htables;

static int htables_reserve(htables* t, size_t extra) {
  if (t->cap - t->size >= extra) return 1;
  size_t cap = t->cap ? t->cap : 4096;
  while (cap - t->size < extra) {
    if (cap > ((size_t)1 << 30)) return 0;
    cap *= 2;
  }
  hentry* e = (hentry*)realloc(t->entries, cap * sizeof(hentry));
  if (e == NULL) return 0;
  t->entries = e;
  t->cap = cap;
  return 1;
}

static inline uint32_t reverse_bits(uint32_t code, int len) {
  uint32_t r = 0;
  for (int i = 0; i < len; ++i) r |= ((code >> i) & 1) << (len - 1 - i);
  return r;
}

/* Builds the table for code lengths `len[0..n)` (n <= MAX_GREEN_ALPHABET) with
 * at most max_root root bits, appended to `t`. Returns 0 for invalid codes:
 * no symbols, over-subscribed, or incomplete with more than one symbol. A
 * single symbol takes zero bits. Returns -1 when out of memory. */
static int build_code(htables* t, const uint8_t* len, int n, int max_root,
                      hcode* out) {
  int count[MAX_CODE_LENGTH + 1] = {0};
  int max_len = 0, num = 0, single = 0;
  for (int s = 0; s < n; ++s) {
    if (len[s] == 0) continue;
    ++count[len[s]];
    ++num;
    single = s;
    if (len[s] > max_len) max_len = len[s];
  }
  if (num == 0) return 0;
  out->offset = (uint32_t)t->size;
  if (num == 1) {
    if (!htables_reserve(t, 1)) return -1;
    t->entries[t->size++] = (hentry){(uint16_t)single, 0, 0};
    out->root_bits = 0;
    return 1;
  }
  int left = 1;
  for (int l = 1; l <= MAX_CODE_LENGTH; ++l) {
    left = (left << 1) - count[l];
    if (left < 0) return 0;
  }
  if (left != 0) return 0;

  /* Symbols sorted by code length, then symbol: canonical code order. */
  uint16_t sorted[MAX_GREEN_ALPHABET];
  int start[MAX_CODE_LENGTH + 2];
  start[1] = 0;
  for (int l = 1; l <= MAX_CODE_LENGTH; ++l) start[l + 1] = start[l] + count[l];
  for (int s = 0; s < n; ++s) {
    if (len[s]) sorted[start[len[s]]++] = (uint16_t)s;
  }

  const int root = max_len < max_root ? max_len : max_root;
  /* Sizes of the second-level tables, in code order (as zlib's
   * inflate_table): a table for a new root prefix gets enough index bits for
   * the codes that share the prefix. First pass: total size. */
  size_t total = (size_t)1 << root;
  {
    int remaining[MAX_CODE_LENGTH + 1];
    memcpy(remaining, count, sizeof(remaining));
    uint32_t code = 0;
    int prev_len = 0;
    uint32_t prefix = UINT32_MAX;
    for (int i = 0; i < num; ++i) {
      const int l = len[sorted[i]];
      code <<= (l - prev_len);
      prev_len = l;
      if (l > root) {
        const uint32_t p = code >> (l - root);
        if (p != prefix) {
          prefix = p;
          int sub = l - root, avail = 1 << sub;
          while (sub + root < max_len) {
            avail -= remaining[sub + root];
            if (avail <= 0) break;
            ++sub;
            avail <<= 1;
          }
          total += (size_t)1 << sub;
        }
      }
      --remaining[l];
      ++code;
    }
  }
  /* Entries address second-level tables with 16 bits. */
  if (total > 0xffff) return 0;
  if (!htables_reserve(t, total)) return -1;
  hentry* tab = t->entries + t->size;
  out->root_bits = (uint8_t)root;

  /* Second pass: fill. */
  int remaining[MAX_CODE_LENGTH + 1];
  memcpy(remaining, count, sizeof(remaining));
  uint32_t code = 0;
  int prev_len = 0;
  uint32_t prefix = UINT32_MAX;
  size_t next_sub = (size_t)1 << root;
  size_t sub_offset = 0;
  int sub_bits = 0;
  for (int i = 0; i < num; ++i) {
    const int s = sorted[i];
    const int l = len[s];
    code <<= (l - prev_len);
    prev_len = l;
    const uint32_t key = reverse_bits(code, l);
    if (l <= root) {
      for (uint32_t k = key; k < (1u << root); k += 1u << l) {
        tab[k] = (hentry){(uint16_t)s, (uint8_t)l, 0};
      }
    } else {
      const uint32_t p = code >> (l - root);
      if (p != prefix) {
        prefix = p;
        int sub = l - root, avail = 1 << sub;
        while (sub + root < max_len) {
          avail -= remaining[sub + root];
          if (avail <= 0) break;
          ++sub;
          avail <<= 1;
        }
        sub_offset = next_sub;
        sub_bits = sub;
        next_sub += (size_t)1 << sub;
        tab[key & ((1u << root) - 1)] =
            (hentry){(uint16_t)sub_offset, (uint8_t)root, (uint8_t)sub};
      }
      const int sl = l - root;
      if (sl > sub_bits) return 0; /* cannot happen for complete codes */
      for (uint32_t k = key >> root; k < (1u << sub_bits); k += 1u << sl) {
        tab[sub_offset + k] = (hentry){(uint16_t)s, (uint8_t)sl, 0};
      }
    }
    --remaining[l];
    ++code;
  }
  t->size += total;
  return 1;
}

/* Decodes a symbol; at least 15 bits must be available. */
static inline int read_symbol(const hentry* tab, int root_bits,
                              bitreader* br) {
  hentry e = tab[br_peek(br, root_bits)];
  if (e.sub) {
    br_skip(br, root_bits);
    e = tab[e.value + br_peek(br, e.sub)];
  }
  br_skip(br, e.bits);
  return e.value;
}

/* ------------------------------------------------------------------------ */
/* Reading prefix codes */

static const uint8_t kCodeLengthOrder[NUM_CODE_LENGTH_CODES] = {
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

/* Reads the code lengths of a prefix code for an alphabet of n symbols.
 * Returns 0 on invalid data. */
static int read_code_lengths(bitreader* br, htables* scratch, int n,
                             uint8_t* len) {
  memset(len, 0, (size_t)n);
  if (br_read(br, 1)) { /* simple code: one or two symbols of 1 bit */
    const int num_symbols = (int)br_read(br, 1) + 1;
    const int first_bits = br_read(br, 1) ? 8 : 1;
    int s = (int)br_read(br, first_bits);
    if (s < n) len[s] = 1;
    if (num_symbols == 2) {
      s = (int)br_read(br, 8);
      if (s < n) len[s] = 1;
    }
    return 1;
  }
  uint8_t cl_len[NUM_CODE_LENGTH_CODES] = {0};
  const int num_codes = (int)br_read(br, 4) + 4;
  for (int i = 0; i < num_codes; ++i) {
    cl_len[kCodeLengthOrder[i]] = (uint8_t)br_read(br, 3);
  }
  scratch->size = 0;
  hcode cl;
  const int r = build_code(scratch, cl_len, NUM_CODE_LENGTH_CODES, 7, &cl);
  if (r <= 0) return r;
  const hentry* cl_tab = scratch->entries + cl.offset;

  int max_symbol = n;
  if (br_read(br, 1)) {
    const int bits = 2 + 2 * (int)br_read(br, 3);
    max_symbol = 2 + (int)br_read(br, bits);
    if (max_symbol > n) return 0;
  }
  int prev_len = 8;
  int s = 0;
  while (s < n) {
    if (max_symbol-- == 0) break;
    br_refill(br);
    const int c = read_symbol(cl_tab, cl.root_bits, br);
    if (c < 16) {
      len[s++] = (uint8_t)c;
      if (c != 0) prev_len = c;
    } else {
      static const int kExtraBits[3] = {2, 3, 7};
      static const int kOffset[3] = {3, 3, 11};
      const int repeat = (int)br_take(br, kExtraBits[c - 16]) + kOffset[c - 16];
      if (s + repeat > n) return 0;
      memset(len + s, c == 16 ? prev_len : 0, (size_t)repeat);
      s += repeat;
    }
  }
  return !br_overrun(br);
}

/* ------------------------------------------------------------------------ */
/* Decoder state */

/* Pair tables decode green and red with one lookup of the next PAIR_BITS
 * bits. An entry holds the bits to consume (bits 0-5, so that the entry
 * itself can be a 64-bit shift count), flags, and green (12 bits) from bit
 * 8, with red from bit 16 for PAIR_RED, so that a literal's green and red
 * are in place. PAIR_GREEN: green fits; PAIR_LITERAL: green is a literal
 * (so its bits 16 and up are clear); PAIR_RED: red fits too; PAIR_CACHE:
 * green is a color cache code. Without PAIR_GREEN, green takes the normal
 * table. */
#define PAIR_BITS 11
#define MANY_PAIR_BITS 10 /* with many groups */
#define PAIR_LEN_MASK 63u
#define PAIR_GREEN (1u << 6)
#define PAIR_LITERAL (1u << 7)
#define PAIR_RED (1u << 24)
#define PAIR_CACHE (1u << 25)
/* Groups covering fewer pixels decode without a pair table. */
#define PAIR_MIN_PIXELS 8192

typedef struct {
  hcode codes[CODES_PER_GROUP];
  /* Set by finish_groups(). */
  const hentry* tab[CODES_PER_GROUP];
  uint32_t mask[CODES_PER_GROUP]; /* root table index masks */
  const uint32_t* pair_tab;       /* or NULL */
  uint32_t pair_mask;             /* pair table index mask */
  int alpha_trivial;  /* alpha has a single symbol: alpha_bits */
  uint32_t alpha_bits;
} group;

typedef struct {
  int type;
  int width;           /* of the rows the inverse transform outputs */
  int bits;            /* tile size (predictor, cross color), or color
                          indexing's bundling (pixels per byte = 1 << bits) */
  int xsize;           /* sub-image width in tiles */
  uint32_t* data;      /* sub-image, or the 256-entry color table */
} transform;

struct wll_decoder {
  wll_mem mem;
  htables tables;
  htables scratch;
  group* groups;
  size_t groups_cap;
  uint32_t* pairs;     /* pair tables */
  size_t pairs_cap;
  uint32_t* rows;     /* row buffers for the inverse transforms */
  size_t rows_cap;
  uint32_t cache[1 << MAX_CACHE_BITS]; /* color cache */
};

/* Color cache of the image stream being decoded. Decoded pixels are
 * inserted lazily, in order, before each lookup and after each run. */
typedef struct {
  uint32_t* table; /* NULL without a color cache */
  int shift;       /* 32 - cache bits */
  size_t cached;   /* pixels before this index are inserted */
} colorcache;

static inline void cache_catch_up(colorcache* cc, const uint32_t* data,
                                  size_t pos) {
  for (size_t i = cc->cached; i < pos; ++i) {
    cc->table[(0x1e35a7bdu * data[i]) >> cc->shift] = data[i];
  }
  cc->cached = pos;
}

typedef struct {
  wll_decoder* dec;
  bitreader br;
  int width, height;
  int xsize;           /* width of the coded image (less with bundling) */
  transform transforms[4];
  int num_transforms;
  /* Main image prefix codes. */
  uint32_t* meta;      /* group per tile, or NULL for one group */
  int meta_bits, meta_xsize;
  int num_groups;
  uint32_t* pixels;    /* decoded main image, before inverse transforms */
  /* Output. */
  uint8_t* rgba;
  size_t stride;
  uint32_t* tmp;       /* one row */
  uint32_t* tmp2;      /* one row, for color indexing's output */
  uint32_t* pred_rows; /* two rows of predictor output */
  int next_row;        /* first row not yet written */
  colorcache cc;
} state;

/* Reads an image stream's color cache bits (0 for none); -1 if invalid. */
static int read_cache_bits(bitreader* br) {
  if (!br_read(br, 1)) return 0;
  const int bits = (int)br_read(br, 4);
  return bits >= 1 && bits <= MAX_CACHE_BITS ? bits : -1;
}

/* Starts an image stream's (empty) color cache of 1 << bits entries. */
static void cache_start(state* st, int bits) {
  st->cc.table = NULL;
  st->cc.cached = 0;
  if (bits == 0) return;
  st->cc.table = st->dec->cache;
  st->cc.shift = 32 - bits;
  memset(st->cc.table, 0, sizeof(uint32_t) << bits);
}

/* Reads the prefix code groups of an image stream with a color cache of
 * 1 << cache_bits entries (or none for 0) into dec->groups. With `used`
 * (tiles per group index), groups not used are validated but not kept.
 * Returns 1, 0 (invalid data) or -1 (out of memory). */
static int read_groups(state* st, int num_groups, const uint32_t* used,
                       int max_root, int cache_bits) {
  wll_decoder* dec = st->dec;
  /* Each code takes at least 4 bits. */
  if ((uint64_t)num_groups * CODES_PER_GROUP * 4 >
      (uint64_t)(st->br.size + 8) * 8) {
    return 0;
  }
  if ((size_t)num_groups > dec->groups_cap) {
    group* g = (group*)realloc(dec->groups, (size_t)num_groups * sizeof(group));
    if (g == NULL) return -1;
    dec->groups = g;
    dec->groups_cap = (size_t)num_groups;
  }
  uint8_t len[MAX_GREEN_ALPHABET];
  for (int i = 0; i < num_groups; ++i) {
    const size_t mark = dec->tables.size;
    for (int c = 0; c < CODES_PER_GROUP; ++c) {
      const int n = kAlphabetSize[c] +
                    (c == GREEN && cache_bits > 0 ? 1 << cache_bits : 0);
      int r = read_code_lengths(&st->br, &dec->scratch, n, len);
      if (r <= 0) return r;
      r = build_code(&dec->tables, len, n, max_root, &dec->groups[i].codes[c]);
      if (r <= 0) return r;
    }
    if (used != NULL && !used[i]) dec->tables.size = mark;
  }
  return 1;
}

/* Decodes a symbol from the low bits of `bits` (zero beyond what is known);
 * *len receives its code length. */
static int peek_symbol(const hentry* tab, int root_bits, uint32_t bits,
                       int* len) {
  const hentry e = tab[bits & ((1u << root_bits) - 1)];
  if (e.sub) {
    const hentry e2 = tab[e.value + ((bits >> root_bits) & ((1u << e.sub) - 1))];
    *len = root_bits + e2.bits;
    return e2.value;
  }
  *len = e.bits;
  return e.value;
}

/* Fills a pair table indexed by `bits` <= PAIR_BITS bits. */
static void build_pairs(const group* g, uint32_t* pairs, int bits) {
  for (uint32_t idx = 0; idx < (1u << bits); ++idx) {
    int gl, rl;
    const int green = peek_symbol(g->tab[GREEN], g->codes[GREEN].root_bits,
                                  idx, &gl);
    uint32_t e = 0;
    /* A decoded length within `bits` only used known bits. */
    if (gl <= bits) {
      e = (uint32_t)green << 8 | (uint32_t)gl | PAIR_GREEN;
      if (green >= GREEN_ALPHABET) e |= PAIR_CACHE;
      if (green < NUM_LITERALS) {
        e |= PAIR_LITERAL;
        const int red = peek_symbol(g->tab[RED], g->codes[RED].root_bits,
                                    idx >> gl, &rl);
        if (gl + rl <= bits) {
          e = (uint32_t)red << 16 | (uint32_t)green << 8 | (uint32_t)(gl + rl) |
              PAIR_GREEN | PAIR_LITERAL | PAIR_RED;
        }
      }
    }
    pairs[idx] = e;
  }
}

/* Prepares groups [0, num_groups) for decoding: table pointers, trivial
 * alpha, and pair tables of pair_bits for groups of at least
 * PAIR_MIN_PIXELS pixels (`pixels` per group, or NULL for none). Groups
 * without pixels have no tables (read_groups() dropped them) and are left
 * out. Returns 0 when out of memory. */
static int finish_groups(wll_decoder* dec, int num_groups,
                         const uint64_t* pixels, int pair_bits) {
  size_t num_pairs = 0;
  for (int i = 0; i < num_groups; ++i) {
    num_pairs += pixels != NULL && pixels[i] >= PAIR_MIN_PIXELS;
  }
  const size_t need = num_pairs << pair_bits;
  if (need > dec->pairs_cap) {
    free(dec->pairs);
    dec->pairs = (uint32_t*)malloc(need * sizeof(uint32_t));
    dec->pairs_cap = dec->pairs != NULL ? need : 0;
    if (dec->pairs == NULL) return 0;
  }
  size_t next = 0;
  for (int i = 0; i < num_groups; ++i) {
    group* g = &dec->groups[i];
    if (pixels != NULL && pixels[i] == 0) {
      memset(g, 0, sizeof(*g));
      continue;
    }
    for (int c = 0; c < CODES_PER_GROUP; ++c) {
      g->tab[c] = dec->tables.entries + g->codes[c].offset;
      g->mask[c] = (1u << g->codes[c].root_bits) - 1;
    }
    g->alpha_trivial = g->codes[ALPHA].root_bits == 0;
    g->alpha_bits = (uint32_t)g->tab[ALPHA][0].value << 24;
    g->pair_tab = NULL;
    g->pair_mask = (1u << pair_bits) - 1;
    if (pixels != NULL && pixels[i] >= PAIR_MIN_PIXELS) {
      build_pairs(g, dec->pairs + next, pair_bits);
      g->pair_tab = dec->pairs + next;
      next += (size_t)1 << pair_bits;
    }
  }
  return 1;
}

static inline uint32_t prefix_value(bitreader* br, int sym) {
  if (sym < 4) return (uint32_t)sym + 1;
  const int extra = (sym - 2) >> 1;
  const uint32_t offset = (uint32_t)(2 + (sym & 1)) << extra;
  return offset + br_take(br, extra) + 1;
}

static const uint8_t kCodeToPlane[120] = {
    0x18, 0x07, 0x17, 0x19, 0x28, 0x06, 0x27, 0x29, 0x16, 0x1a, 0x26, 0x2a,
    0x38, 0x05, 0x37, 0x39, 0x15, 0x1b, 0x36, 0x3a, 0x25, 0x2b, 0x48, 0x04,
    0x47, 0x49, 0x14, 0x1c, 0x35, 0x3b, 0x46, 0x4a, 0x24, 0x2c, 0x58, 0x45,
    0x4b, 0x34, 0x3c, 0x03, 0x57, 0x59, 0x13, 0x1d, 0x56, 0x5a, 0x23, 0x2d,
    0x44, 0x4c, 0x55, 0x5b, 0x33, 0x3d, 0x68, 0x02, 0x67, 0x69, 0x12, 0x1e,
    0x66, 0x6a, 0x22, 0x2e, 0x54, 0x5c, 0x43, 0x4d, 0x65, 0x6b, 0x32, 0x3e,
    0x78, 0x01, 0x77, 0x79, 0x53, 0x5d, 0x11, 0x1f, 0x64, 0x6c, 0x42, 0x4e,
    0x76, 0x7a, 0x21, 0x2f, 0x75, 0x7b, 0x31, 0x3f, 0x63, 0x6d, 0x52, 0x5e,
    0x00, 0x74, 0x7c, 0x41, 0x4f, 0x10, 0x20, 0x62, 0x6e, 0x30, 0x73, 0x7d,
    0x51, 0x5f, 0x40, 0x72, 0x7e, 0x61, 0x6f, 0x50, 0x71, 0x7f, 0x60, 0x70};

static inline size_t plane_code_to_distance(int xsize, uint32_t code) {
  if (code > 120) return code - 120;
  const int dist_code = kCodeToPlane[code - 1];
  const int yoffset = dist_code >> 4;
  const int xoffset = 8 - (dist_code & 0xf);
  const long dist = (long)yoffset * xsize + xoffset;
  return dist >= 1 ? (size_t)dist : 1;
}

static void process_rows(state* st, int end_row);

/* read_symbol with the root mask precomputed. */
static inline int read_sym(const hentry* tab, uint32_t mask, int root_bits,
                           bitreader* br) {
  hentry e = tab[br->buf & mask];
  if (e.sub) {
    br_skip(br, root_bits);
    e = tab[e.value + br_peek(br, e.sub)];
  }
  br_skip(br, e.bits);
  return e.value;
}

/* LZ77 copy of `length` pixels to dst from dist pixels back (dist >= 1),
 * pixel by pixel in effect: the source may overlap the destination. */
static inline void copy_pixels(uint32_t* dst, size_t dist, size_t length) {
  if (dist == 1) {
    const uint32_t v = dst[-1];
    for (size_t k = 0; k < length; ++k) dst[k] = v;
    return;
  }
  /* Chunks of at most dist pixels never overlap their source. */
  while (length > 0) {
    const size_t chunk = length < dist ? length : dist;
    memcpy(dst, dst - dist, chunk * sizeof(*dst));
    dst += chunk;
    length -= chunk;
  }
}

/* Decodes one literal or copy at data[*pos] with group g's tables. `e` is
 * the pair table entry for the next bits (0 without a pair table), not one
 * with PAIR_LITERAL; if it has PAIR_GREEN, green was decoded but its bits
 * are not consumed yet. Returns 0 on invalid data. */
static inline __attribute__((always_inline)) int decode_item_impl(
    const group* g, uint32_t e, bitreader* br, uint32_t* data, size_t* pos,
    size_t n, int xsize, colorcache* cc) {
  br_refill(br);
  int green;
  if (e & PAIR_GREEN) {
    green = (int)((e >> 8) & 0xfff);
    br_skip(br, (int)(e & PAIR_LEN_MASK));
  } else {
    green = read_sym(g->tab[GREEN], g->mask[GREEN], g->codes[GREEN].root_bits,
                     br);
  }
  if (green < NUM_LITERALS) {
    const int red =
        read_sym(g->tab[RED], g->mask[RED], g->codes[RED].root_bits, br);
    br_refill(br);
    const int blue =
        read_sym(g->tab[BLUE], g->mask[BLUE], g->codes[BLUE].root_bits, br);
    const uint32_t alpha =
        g->alpha_trivial
            ? g->alpha_bits
            : (uint32_t)read_sym(g->tab[ALPHA], g->mask[ALPHA],
                                 g->codes[ALPHA].root_bits, br)
                  << 24;
    data[(*pos)++] =
        alpha | (uint32_t)red << 16 | (uint32_t)green << 8 | (uint32_t)blue;
    return 1;
  }
  if (green >= GREEN_ALPHABET) {
    /* Color cache codes only exist with a cache, within its size. */
    if (cc->table == NULL) return 0;
    cache_catch_up(cc, data, *pos);
    data[(*pos)++] = cc->table[green - GREEN_ALPHABET];
    return 1;
  }
  const size_t length = prefix_value(br, green - NUM_LITERALS);
  br_refill(br);
  const int dist_sym =
      read_sym(g->tab[DIST], g->mask[DIST], g->codes[DIST].root_bits, br);
  const size_t dist = plane_code_to_distance(xsize, prefix_value(br, dist_sym));
  const size_t p = *pos;
  if (dist > p || length > n - p) return 0;
  copy_pixels(data + p, dist, length);
  *pos = p + length;
  return 1;
}

typedef int (*decode_item_fn)(const group*, uint32_t, bitreader*, uint32_t*,
                              size_t*, size_t, int, colorcache*);

static __attribute__((noinline)) int decode_item(const group* g, uint32_t e,
                                                 bitreader* br, uint32_t* data,
                                                 size_t* pos, size_t n,
                                                 int xsize, colorcache* cc) {
  return decode_item_impl(g, e, br, data, pos, n, xsize, cc);
}

#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
__attribute__((noinline, target("bmi2"))) static int decode_item_bmi2(
    const group* g, uint32_t e, bitreader* br, uint32_t* data, size_t* pos,
    size_t n, int xsize, colorcache* cc) {
  return decode_item_impl(g, e, br, data, pos, n, xsize, cc);
}
#endif

/* Decodes data[*pos, end) in a group with a pair table. Literals whose
 * green the pair table decodes (and with_cache, color cache codes) take
 * this loop, with the bit reader in local variables; other items go through
 * `item`. with_alpha: alpha is not constant. With a color cache
 * (with_cache), pixels are inserted as they are decoded. Returns 0 on
 * invalid data. */
static inline __attribute__((always_inline)) int fast_run(
    const group* g, bitreader* br_out, uint32_t* data, size_t* pos_out,
    size_t end, size_t n, int xsize, colorcache* cc, decode_item_fn item,
    const int with_cache, const int with_alpha) {
  const uint32_t* pair_tab = g->pair_tab;
  const uint32_t pair_mask = g->pair_mask;
  const hentry *red_tab = g->tab[RED], *blue_tab = g->tab[BLUE];
  const hentry* alpha_tab = g->tab[ALPHA];
  const uint32_t red_mask = g->mask[RED], blue_mask = g->mask[BLUE];
  const uint32_t alpha_mask = g->mask[ALPHA];
  const int red_root = g->codes[RED].root_bits;
  const int blue_root = g->codes[BLUE].root_bits;
  const int alpha_root = g->codes[ALPHA].root_bits;
  const uint32_t alpha_bits = g->alpha_bits;
  uint32_t* cache = cc->table;
  const int cache_shift = cc->shift;
  bitreader br = *br_out;
  const uint8_t* in = br.data;
  const size_t in_fast_end = br.size >= 8 ? br.size - 8 : 0;
  uint64_t buf = br.buf;
  int nbits = br.nbits;
  size_t in_pos = br.pos;
  size_t pos = *pos_out;
  if (with_cache) cache_catch_up(cc, data, pos);
#define REFILL()                                         \
  do {                                                   \
    if (in_pos <= in_fast_end && br.size >= 8) {         \
      buf |= load64le(in + in_pos) << nbits;             \
      in_pos += (size_t)((63 - nbits) >> 3);             \
      nbits |= 56;                                       \
    } else {                                             \
      br.buf = buf, br.nbits = nbits, br.pos = in_pos;   \
      br_refill(&br);                                    \
      buf = br.buf, nbits = br.nbits, in_pos = br.pos;   \
    }                                                    \
  } while (0)
#define CACHE_INSERT(px) \
  if (with_cache) cache[(0x1e35a7bdu * (px)) >> cache_shift] = (px)
  /* At the top of the loop at least PAIR_BITS bits are buffered: since the
   * last refill to 56 or more bits, at most PAIR_BITS + 2 * 15 were
   * consumed. So the pair lookup goes first and the refill, off the
   * critical path, after. */
  REFILL();
  while (pos < end) {
    const uint32_t e = pair_tab[buf & pair_mask];
    REFILL();
    if (e & PAIR_LITERAL) {
      const int len = (int)(e & PAIR_LEN_MASK);
      buf >>= len;
      nbits -= len;
      uint32_t green_red = e & 0x00ffff00u;
      if (!(e & PAIR_RED)) {
        hentry r = red_tab[buf & red_mask];
        if (r.sub) {
          buf >>= red_root;
          nbits -= red_root;
          r = red_tab[r.value + (buf & ((1u << r.sub) - 1))];
        }
        buf >>= r.bits;
        nbits -= r.bits;
        green_red |= (uint32_t)r.value << 16;
        REFILL(); /* for blue */
      }
      /* After a pair, blue fits the bits left from the first refill. */
      hentry b = blue_tab[buf & blue_mask];
      if (b.sub) {
        buf >>= blue_root;
        nbits -= blue_root;
        b = blue_tab[b.value + (buf & ((1u << b.sub) - 1))];
      }
      buf >>= b.bits;
      nbits -= b.bits;
      uint32_t alpha = alpha_bits;
      if (with_alpha) {
        /* Pair, blue and alpha take at most 41 of the refilled bits. */
        hentry a = alpha_tab[buf & alpha_mask];
        if (a.sub) {
          buf >>= alpha_root;
          nbits -= alpha_root;
          a = alpha_tab[a.value + (buf & ((1u << a.sub) - 1))];
        }
        buf >>= a.bits;
        nbits -= a.bits;
        alpha = (uint32_t)a.value << 24;
      }
      const uint32_t px = alpha | green_red | b.value;
      data[pos++] = px;
      CACHE_INSERT(px);
    } else if (with_cache && (e & PAIR_CACHE)) {
      const int len = (int)(e & PAIR_LEN_MASK);
      buf >>= len;
      nbits -= len;
      const uint32_t px = cache[((e >> 8) & 0xfff) - GREEN_ALPHABET];
      data[pos++] = px;
      CACHE_INSERT(px);
    } else {
      br.buf = buf, br.nbits = nbits, br.pos = in_pos;
      if (with_cache) cc->cached = pos;
      if (!item(g, e, &br, data, &pos, n, xsize, cc)) return 0;
      if (with_cache) cache_catch_up(cc, data, pos);
      buf = br.buf, nbits = br.nbits, in_pos = br.pos;
      REFILL();
    }
  }
#undef REFILL
#undef CACHE_INSERT
  if (with_cache) cc->cached = pos;
  br.buf = buf, br.nbits = nbits, br.pos = in_pos;
  *br_out = br;
  *pos_out = pos;
  return 1;
}

/* Decodes the entropy-coded pixels of an image with the groups in
 * dec->groups. For the main image finished rows are written out as decoding
 * goes. Returns 0 on invalid data.
 *
 * Pixels are decoded in runs over tiles of the meta image with the same
 * group; a copy past the run's end ends it. In groups with a pair
 * table, literals take a tight loop (fast_run()); everything else goes
 * through decode_item(). */
static inline __attribute__((always_inline)) int decode_pixels_impl(
    state* st, uint32_t* data, int xsize, int ysize, int is_main,
    decode_item_fn item) {
  bitreader br = st->br;
  const group* groups = st->dec->groups;
  const uint32_t* meta = is_main ? st->meta : NULL;
  const int meta_bits = st->meta_bits, meta_xsize = st->meta_xsize;
  const size_t n = (size_t)xsize * (size_t)ysize;
  size_t pos = 0;
  int col = 0, row = 0;
  while (pos < n) {
    const group* g = &groups[0];
    int run = xsize - col;
    if (meta != NULL) {
      /* The run extends over the following tiles of the same group. */
      const uint32_t* tiles =
          meta + (size_t)(row >> meta_bits) * (size_t)meta_xsize;
      int t = col >> meta_bits;
      const uint32_t gi = tiles[t];
      g = &groups[gi];
      while (++t < meta_xsize && tiles[t] == gi) {
      }
      const int run_end = t << meta_bits;
      if (run_end - col < run) run = run_end - col;
    }
    const size_t start = pos, end = pos + (size_t)run;
    if (g->pair_tab != NULL) {
      const int cache = st->cc.table != NULL, alpha = !g->alpha_trivial;
      int ok;
      if (cache) {
        ok = alpha ? fast_run(g, &br, data, &pos, end, n, xsize, &st->cc, item,
                              1, 1)
                   : fast_run(g, &br, data, &pos, end, n, xsize, &st->cc, item,
                              1, 0);
      } else {
        ok = alpha ? fast_run(g, &br, data, &pos, end, n, xsize, &st->cc, item,
                              0, 1)
                   : fast_run(g, &br, data, &pos, end, n, xsize, &st->cc, item,
                              0, 0);
      }
      if (!ok) return 0;
    } else {
      while (pos < end) {
        if (!item(g, 0, &br, data, &pos, n, xsize, &st->cc)) return 0;
      }
    }
    if (st->cc.table != NULL) cache_catch_up(&st->cc, data, pos);
    const size_t c = (size_t)col + (pos - start);
    if (c < (size_t)xsize) {
      col = (int)c;
    } else {
      row += (int)(c / (size_t)xsize);
      col = (int)(c % (size_t)xsize);
      if (br_overrun(&br)) return 0;
      if (is_main) process_rows(st, row);
    }
  }
  st->br = br;
  return !br_overrun(&br);
}

#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
/* With BMI2 the many variable shifts take one instruction each. */
__attribute__((target("bmi2"))) static int decode_pixels_bmi2(
    state* st, uint32_t* data, int xsize, int ysize, int is_main) {
  return decode_pixels_impl(st, data, xsize, ysize, is_main, decode_item_bmi2);
}
#endif

static int decode_pixels(state* st, uint32_t* data, int xsize, int ysize,
                         int is_main) {
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
  if (__builtin_cpu_supports("bmi2")) {
    return decode_pixels_bmi2(st, data, xsize, ysize, is_main);
  }
#endif
  return decode_pixels_impl(st, data, xsize, ysize, is_main, decode_item);
}

/* Reads and decodes an image stream that is not the main image (no
 * transforms, one group) of xsize * ysize pixels into a new buffer. Returns
 * a wll_status. */
static wll_status decode_sub_image(state* st, int xsize, int ysize,
                                   uint32_t** out) {
  const int cache_bits = read_cache_bits(&st->br);
  if (cache_bits < 0) return WLL_ERR_INVALID_DATA;
  const int r = read_groups(st, 1, NULL, 8, cache_bits);
  if (r == 0) return WLL_ERR_INVALID_DATA;
  if (r < 0 || !finish_groups(st->dec, 1, NULL, PAIR_BITS)) {
    return WLL_ERR_OUT_OF_MEMORY;
  }
  uint32_t* data = (uint32_t*)malloc((size_t)xsize * (size_t)ysize * 4);
  if (data == NULL) return WLL_ERR_OUT_OF_MEMORY;
  /* Sub-images use the first entries of the table array; the main image's
   * groups are read after all sub-images. */
  cache_start(st, cache_bits);
  const int ok = decode_pixels(st, data, xsize, ysize, 0);
  cache_start(st, 0);
  if (!ok) {
    free(data);
    return WLL_ERR_INVALID_DATA;
  }
  st->dec->tables.size = 0;
  *out = data;
  return WLL_OK;
}

/* ------------------------------------------------------------------------ */
/* Inverse transforms */

static inline uint32_t add_pixels(uint32_t a, uint32_t b) {
  const uint32_t ag = (a & 0xff00ff00u) + (b & 0xff00ff00u);
  const uint32_t rb = (a & 0x00ff00ffu) + (b & 0x00ff00ffu);
  return (ag & 0xff00ff00u) | (rb & 0x00ff00ffu);
}

static inline uint32_t average2(uint32_t a, uint32_t b) {
  return (((a ^ b) & 0xfefefefeu) >> 1) + (a & b);
}

static inline int ch(uint32_t v, int shift) { return (int)((v >> shift) & 0xff); }

static inline uint32_t clip255(int v) {
  return v < 0 ? 0u : v > 255 ? 255u : (uint32_t)v;
}

static inline uint32_t clamped_add_subtract_full(uint32_t a, uint32_t b,
                                                 uint32_t c) {
  uint32_t r = 0;
  for (int s = 0; s < 32; s += 8) {
    r |= clip255(ch(a, s) + ch(b, s) - ch(c, s)) << s;
  }
  return r;
}

static inline uint32_t clamped_add_subtract_half(uint32_t c0, uint32_t c1,
                                                 uint32_t c2) {
  const uint32_t ave = average2(c0, c1);
  uint32_t r = 0;
  for (int s = 0; s < 32; s += 8) {
    const int a = ch(ave, s);
    r |= clip255(a + (a - ch(c2, s)) / 2) << s;
  }
  return r;
}

static inline uint32_t select_pred(uint32_t a, uint32_t b, uint32_t c) {
  int pa_minus_pb = 0;
  for (int s = 0; s < 32; s += 8) {
    pa_minus_pb += abs(ch(b, s) - ch(c, s)) - abs(ch(a, s) - ch(c, s));
  }
  return pa_minus_pb <= 0 ? a : b;
}

/* Prediction for x > 0, y > 0; top points at the pixel above. */
static inline uint32_t predict(int mode, uint32_t left, const uint32_t* top) {
  switch (mode) {
    case 1: return left;
    case 2: return top[0];
    case 3: return top[1];
    case 4: return top[-1];
    case 5: return average2(average2(left, top[1]), top[0]);
    case 6: return average2(left, top[-1]);
    case 7: return average2(left, top[0]);
    case 8: return average2(top[-1], top[0]);
    case 9: return average2(top[0], top[1]);
    case 10: return average2(average2(left, top[-1]), average2(top[0], top[1]));
    case 11: return select_pred(top[0], left, top[-1]);
    case 12: return clamped_add_subtract_full(left, top[0], top[-1]);
    case 13: return clamped_add_subtract_half(left, top[0], top[-1]);
    default: return ARGB_BLACK; /* 0, and 14 and 15 as libwebp */
  }
}

/* Mode 11 (select) for x in [x0, x1), x0 >= 1. Serial through the left
 * pixel, which stays in a vector register; psadbw sums the channel
 * differences. */
static void select_run(const uint32_t* in, const uint32_t* up, uint32_t* out,
                       int x0, int x1) {
#if defined(__SSE2__)
  __m128i left = _mm_cvtsi32_si128((int)out[x0 - 1]);
  for (int x = x0; x < x1; ++x) {
    const __m128i top = _mm_cvtsi32_si128((int)up[x]);
    const __m128i top_left = _mm_cvtsi32_si128((int)up[x - 1]);
    const __m128i res = _mm_cvtsi32_si128((int)in[x]);
    /* Left when sum |L - TL| > sum |T - TL|. The residual is added to both
     * candidates, which keeps the addition off the serial path. */
    const __m128i use_left = _mm_cmpgt_epi32(_mm_sad_epu8(left, top_left),
                                             _mm_sad_epu8(top, top_left));
    left = _mm_or_si128(_mm_and_si128(use_left, _mm_add_epi8(left, res)),
                        _mm_andnot_si128(use_left, _mm_add_epi8(top, res)));
    out[x] = (uint32_t)_mm_cvtsi128_si32(left);
  }
#else
  for (int x = x0; x < x1; ++x) {
    out[x] = add_pixels(in[x], select_pred(up[x], out[x - 1], up[x - 1]));
  }
#endif
}

/* Inverse predictor `mode` for x in [x0, x1), 1 <= x0, x1 < width (so that
 * the top-right pixel is up[x + 1]). */
static void predict_run_c(int mode, const uint32_t* in, const uint32_t* up,
                          uint32_t* out, int x0, int x1) {
  uint32_t left = out[x0 - 1];
#define FROM_ABOVE(pred)                                              \
  for (int x = x0; x < x1; ++x) out[x] = add_pixels(in[x], (pred)); \
  break
#define FROM_LEFT(pred)                \
  for (int x = x0; x < x1; ++x) {      \
    left = add_pixels(in[x], (pred));  \
    out[x] = left;                     \
  }                                    \
  break
  switch (mode) {
    case 1: FROM_LEFT(left);
    case 2: FROM_ABOVE(up[x]);
    case 3: FROM_ABOVE(up[x + 1]);
    case 4: FROM_ABOVE(up[x - 1]);
    case 5: FROM_LEFT(average2(average2(left, up[x + 1]), up[x]));
    case 6: FROM_LEFT(average2(left, up[x - 1]));
    case 7: FROM_LEFT(average2(left, up[x]));
    case 8: FROM_ABOVE(average2(up[x - 1], up[x]));
    case 9: FROM_ABOVE(average2(up[x], up[x + 1]));
    case 10:
      FROM_LEFT(average2(average2(left, up[x - 1]), average2(up[x], up[x + 1])));
    case 11: select_run(in, up, out, x0, x1); break;
    case 12: FROM_LEFT(clamped_add_subtract_full(left, up[x], up[x - 1]));
    case 13: FROM_LEFT(clamped_add_subtract_half(left, up[x], up[x - 1]));
    default: FROM_ABOVE(ARGB_BLACK); /* 0, and 14 and 15 as libwebp */
  }
#undef FROM_ABOVE
#undef FROM_LEFT
}

#if defined(__SSE2__)
static inline __m128i load_pixel(uint32_t v) {
  return _mm_cvtsi32_si128((int)v);
}

/* average2() per byte: pavgb rounds up, so subtract the lost low bit. */
static inline __m128i avg_floor(__m128i a, __m128i b) {
  return _mm_sub_epi8(_mm_avg_epu8(a, b),
                      _mm_and_si128(_mm_xor_si128(a, b), _mm_set1_epi8(1)));
}
#endif

/* predict_run_c() with SSE2: four pixels at a time for the modes that only
 * use the row above, a running byte sum for mode 1, and one pixel at a time
 * with the left pixel in a register for the other modes. */
static void predict_run(int mode, const uint32_t* in, const uint32_t* up,
                        uint32_t* out, int x0, int x1) {
#if defined(__SSE2__)
  int x = x0;
  switch (mode) {
    case 0: case 2: case 3: case 4: case 8: case 9: case 14: case 15:
      for (; x + 4 <= x1; x += 4) {
        const __m128i t = _mm_loadu_si128((const __m128i*)(up + x));
        const __m128i tl = _mm_loadu_si128((const __m128i*)(up + x - 1));
        const __m128i tr = _mm_loadu_si128((const __m128i*)(up + x + 1));
        __m128i pred;
        switch (mode) {
          case 2: pred = t; break;
          case 3: pred = tr; break;
          case 4: pred = tl; break;
          case 8: pred = avg_floor(tl, t); break;
          case 9: pred = avg_floor(t, tr); break;
          default: pred = _mm_set1_epi32((int)ARGB_BLACK); break;
        }
        _mm_storeu_si128(
            (__m128i*)(out + x),
            _mm_add_epi8(_mm_loadu_si128((const __m128i*)(in + x)), pred));
      }
      break;
    case 1: {
      __m128i carry = _mm_shuffle_epi32(load_pixel(out[x0 - 1]), 0);
      for (; x + 4 <= x1; x += 4) {
        __m128i v = _mm_loadu_si128((const __m128i*)(in + x));
        v = _mm_add_epi8(v, _mm_slli_si128(v, 4));
        v = _mm_add_epi8(v, _mm_slli_si128(v, 8));
        v = _mm_add_epi8(v, carry);
        _mm_storeu_si128((__m128i*)(out + x), v);
        carry = _mm_shuffle_epi32(v, 0xff);
      }
      break;
    }
    case 11:
      select_run(in, up, out, x0, x1);
      return;
    default: {
      const __m128i zero = _mm_setzero_si128();
      __m128i left = load_pixel(out[x0 - 1]);
      for (; x < x1; ++x) {
        const __m128i t = load_pixel(up[x]), tl = load_pixel(up[x - 1]);
        const __m128i tr = load_pixel(up[x + 1]);
        __m128i pred;
        switch (mode) {
          case 5: pred = avg_floor(avg_floor(left, tr), t); break;
          case 6: pred = avg_floor(left, tl); break;
          case 7: pred = avg_floor(left, t); break;
          case 10:
            pred = avg_floor(avg_floor(left, tl), avg_floor(t, tr));
            break;
          case 12: {
            /* Clamp L + T - TL per channel: 16-bit lanes, saturating pack. */
            const __m128i d = _mm_sub_epi16(_mm_unpacklo_epi8(t, zero),
                                            _mm_unpacklo_epi8(tl, zero));
            pred = _mm_packus_epi16(
                _mm_add_epi16(_mm_unpacklo_epi8(left, zero), d), zero);
            break;
          }
          default: {
            /* Mode 13: clamp a + (a - TL) / 2, a = average2(L, T), with
             * division truncating toward zero. */
            const __m128i a = _mm_unpacklo_epi8(avg_floor(left, t), zero);
            const __m128i d = _mm_sub_epi16(a, _mm_unpacklo_epi8(tl, zero));
            const __m128i half =
                _mm_srai_epi16(_mm_sub_epi16(d, _mm_srai_epi16(d, 15)), 1);
            pred = _mm_packus_epi16(_mm_add_epi16(a, half), zero);
            break;
          }
        }
        left = _mm_add_epi8(pred, load_pixel(in[x]));
        out[x] = (uint32_t)_mm_cvtsi128_si32(left);
      }
      return;
    }
  }
  if (x < x1) predict_run_c(mode, in, up, out, x, x1);
#else
  predict_run_c(mode, in, up, out, x0, x1);
#endif
}

/* Inverse predictor of row y: out = in + prediction from out's left and
 * `up` (the previous output row; NULL for y == 0). */
static void inverse_predictor_row(const transform* t, int y, int width,
                                  const uint32_t* in, const uint32_t* up,
                                  uint32_t* out) {
  if (up == NULL) {
    out[0] = add_pixels(in[0], ARGB_BLACK);
    for (int x = 1; x < width; ++x) out[x] = add_pixels(in[x], out[x - 1]);
    return;
  }
  out[0] = add_pixels(in[0], up[0]);
  const uint32_t* modes = t->data + (size_t)(y >> t->bits) * (size_t)t->xsize;
  const int tile = 1 << t->bits;
  for (int x0 = 0; x0 < width; x0 += tile) {
    const int mode = (int)((modes[x0 >> t->bits] >> 8) & 0xf);
    const int x1 = x0 + tile < width ? x0 + tile : width;
    /* The last pixel's top-right is the first pixel of this row. */
    const int run_end = x1 < width ? x1 : width - 1;
    const int run_start = x0 > 0 ? x0 : 1;
    if (run_start < run_end) predict_run(mode, in, up, out, run_start, run_end);
    if (x1 == width && width > 1) {
      const int x = width - 1;
      const uint32_t top[3] = {up[x - 1], up[x], out[0]};
      out[x] = add_pixels(in[x], predict(mode, out[x - 1], top + 1));
    }
  }
}

static inline int color_delta(int8_t pred, int8_t color) {
  return ((int)pred * color) >> 5;
}

WLL_TARGET_CLONES
static void inverse_cross_color_row(const transform* t, int y, int width,
                                    const uint32_t* in, uint32_t* out) {
  const uint32_t* codes = t->data + (size_t)(y >> t->bits) * (size_t)t->xsize;
  const int tile = 1 << t->bits;
  for (int x0 = 0; x0 < width; x0 += tile) {
    const uint32_t m = codes[x0 >> t->bits];
    const int8_t g2r = (int8_t)(m & 0xff), g2b = (int8_t)((m >> 8) & 0xff);
    const int8_t r2b = (int8_t)((m >> 16) & 0xff);
    const int x1 = x0 + tile < width ? x0 + tile : width;
    for (int x = x0; x < x1; ++x) {
      const uint32_t v = in[x];
      const int8_t green = (int8_t)(v >> 8);
      const int red = (int)(((v >> 16) + (uint32_t)color_delta(g2r, green)) &
                            0xff);
      const int blue = (int)((v + (uint32_t)color_delta(g2b, green) +
                              (uint32_t)color_delta(r2b, (int8_t)red)) &
                             0xff);
      out[x] = (v & 0xff00ff00u) | (uint32_t)red << 16 | (uint32_t)blue;
    }
  }
}

WLL_TARGET_CLONES
static void add_green_row(int width, const uint32_t* in, uint32_t* out) {
  for (int x = 0; x < width; ++x) {
    const uint32_t v = in[x];
    const uint32_t g = (v >> 8) & 0xff;
    out[x] = (v & 0xff00ff00u) | (((v & 0x00ff00ffu) + (g << 16 | g)) &
                                  0x00ff00ffu);
  }
}

/* ARGB to RGBA bytes, adding green to red and blue first if add_green. */
WLL_TARGET_CLONES
static void write_rgba_row(int width, const uint32_t* argb, uint8_t* out,
                           int add_green) {
  if (add_green) {
    for (int x = 0; x < width; ++x) {
      const uint32_t v = argb[x];
      const uint32_t g = (v >> 8) & 0xff;
      const uint32_t w = (((v >> 16) + g) & 0xff) | (v & 0xff00u) |
                         ((v + g) & 0xff) << 16 | (v & 0xff000000u);
      memcpy(out + 4 * x, &w, 4);
    }
  } else {
    for (int x = 0; x < width; ++x) {
      const uint32_t v = argb[x];
      const uint32_t w = ((v >> 16) & 0xff) | (v & 0xff00u) |
                         (v & 0xff) << 16 | (v & 0xff000000u);
      memcpy(out + 4 * x, &w, 4);
    }
  }
}

/* Inverse color indexing to `width` pixels: color table entries for the
 * indices in the green bytes of `in`, 1 << t->bits indices per byte. */
static void inverse_color_indexing_row(const transform* t, int width,
                                       const uint32_t* in, uint32_t* out) {
  const uint32_t* colors = t->data;
  if (t->bits == 0) {
    for (int x = 0; x < width; ++x) out[x] = colors[(in[x] >> 8) & 0xff];
    return;
  }
  const int index_bits = 8 >> t->bits;
  const uint32_t index_mask = (1u << index_bits) - 1;
  const int per_byte_mask = (1 << t->bits) - 1;
  for (int x = 0; x < width; ++x) {
    const uint32_t packed = (in[x >> t->bits] >> 8) & 0xff;
    out[x] = colors[(packed >> ((x & per_byte_mask) * index_bits)) &
                    index_mask];
  }
}

/* Writes rows [st->next_row, end_row) of the main image: the inverse
 * transforms in reverse order, then RGBA output. The decoded pixels stay
 * untouched, since later copies refer to them. */
static void process_rows(state* st, int end_row) {
  WLL_TIMER_START(timer);
  for (int y = st->next_row; y < end_row; ++y) {
    const uint32_t* row = st->pixels + (size_t)y * (size_t)st->xsize;
    int add_green = 0;
    for (int i = st->num_transforms - 1; i >= 0; --i) {
      const transform* t = &st->transforms[i];
      const int width = t->width;
      if (t->type == CROSS_COLOR_TRANSFORM) {
        inverse_cross_color_row(t, y, width, row, st->tmp);
        row = st->tmp;
      } else if (t->type == SUBTRACT_GREEN_TRANSFORM) {
        if (i == 0) {
          add_green = 1; /* done while writing the output */
        } else {
          add_green_row(width, row, st->tmp);
          row = st->tmp;
        }
      } else if (t->type == COLOR_INDEXING_TRANSFORM) {
        /* The output is wider than the input, which may be tmp. */
        inverse_color_indexing_row(t, width, row, st->tmp2);
        row = st->tmp2;
      } else {
        /* The predictor's output rows are its inputs for the next row. */
        uint32_t* out = st->pred_rows + (size_t)(y & 1) * (size_t)width;
        const uint32_t* up =
            y > 0 ? st->pred_rows + (size_t)((y - 1) & 1) * (size_t)width
                  : NULL;
        inverse_predictor_row(t, y, width, row, up, out);
        row = out;
      }
    }
    write_rgba_row(st->width, row, st->rgba + (size_t)y * st->stride,
                   add_green);
  }
  st->next_row = end_row;
  WLL_TIMER_LAP(timer, WLL_T_D_ROWS);
}

/* ------------------------------------------------------------------------ */
/* Container and header */

static uint32_t le32(const uint8_t* p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
         (uint32_t)p[3] << 24;
}

/* Finds the VP8L bitstream in a WebP file (or a bare VP8L stream) and reads
 * its dimensions. */
static wll_status parse_header(const uint8_t* data, size_t size,
                               const uint8_t** vp8l, size_t* vp8l_size,
                               int* width, int* height) {
  int canvas_width = 0, canvas_height = 0; /* of a VP8X header */
  if (data == NULL) return WLL_ERR_INVALID_ARG;
  const size_t total = size;
  if (size >= 12 && memcmp(data, "RIFF", 4) == 0 &&
      memcmp(data + 8, "WEBP", 4) == 0) {
    /* As libwebp: the RIFF payload holds at least a chunk header and fits
     * the data, and chunks lie within it. */
    const uint32_t riff_size = le32(data + 4);
    if (riff_size < 12 || riff_size > size - 8) return WLL_ERR_INVALID_DATA;
    size = (size_t)riff_size + 8;
    size_t p = 12;
    if (memcmp(data + p, "VP8X", 4) == 0) {
      /* Extended format: a VP8X chunk, then optional chunks (metadata,
       * which is skipped) up to the image chunk. */
      if (le32(data + p + 4) != 10 || size - p < 18) {
        return WLL_ERR_INVALID_DATA;
      }
      if (le32(data + p + 8) & 0x02) return WLL_ERR_UNSUPPORTED; /* anim */
      canvas_width = 1 + (int)(le32(data + p + 12) & 0xffffff);
      canvas_height = 1 + (int)(le32(data + p + 14) >> 8);
      p += 18;
      for (;;) {
        if (size - p < 8) return WLL_ERR_INVALID_DATA;
        if (memcmp(data + p, "VP8 ", 4) == 0 ||
            memcmp(data + p, "VP8L", 4) == 0) {
          break;
        }
        /* Payloads are padded to an even size. */
        const uint64_t chunk = (8 + (uint64_t)le32(data + p + 4) + 1) & ~1ull;
        if (chunk > size - p) return WLL_ERR_INVALID_DATA;
        p += (size_t)chunk;
      }
    }
    if (total - p < 8) return WLL_ERR_INVALID_DATA;
    if (memcmp(data + p, "VP8 ", 4) == 0) return WLL_ERR_UNSUPPORTED;
    if (memcmp(data + p, "VP8L", 4) == 0) {
      const uint32_t chunk_size = le32(data + p + 4);
      /* With VP8X the chunk counts with its padding, as optional chunks. */
      const uint64_t need = canvas_width != 0
                                ? (8 + (uint64_t)chunk_size + 1) & ~1ull
                                : 8 + (uint64_t)chunk_size;
      if (need > size - p) return WLL_ERR_INVALID_DATA;
      /* Also as libwebp, the bitstream may go on past its chunk, up to the
       * end of the data. */
      data += p + 8;
      size = total - (p + 8);
    } else {
      /* As libwebp: without a chunk header (and VP8X), the rest is a bare
       * bitstream. */
      data += p;
      size = total - p;
    }
  } else if (size >= 8 && memcmp(data, "VP8 ", 4) == 0) {
    return WLL_ERR_UNSUPPORTED;
  } else if (size >= 8 && memcmp(data, "VP8L", 4) == 0) {
    /* As libwebp: an image chunk without a RIFF header. */
    if (le32(data + 4) > size - 8) return WLL_ERR_INVALID_DATA;
    data += 8;
    size -= 8;
  }
  if (size < 5 || data[0] != 0x2f || (data[4] >> 5) != 0) {
    /* Not VP8L. A bare lossy (VP8) key frame has a start code after its
     * 3-byte tag. */
    if (size >= 6 && data[3] == 0x9d && data[4] == 0x01 && data[5] == 0x2a) {
      return WLL_ERR_UNSUPPORTED;
    }
    return WLL_ERR_INVALID_DATA;
  }
  const uint32_t bits = le32(data + 1);
  *width = (int)(bits & 0x3fff) + 1;
  *height = (int)((bits >> 14) & 0x3fff) + 1;
  if (canvas_width != 0 &&
      (canvas_width != *width || canvas_height != *height)) {
    return WLL_ERR_INVALID_DATA;
  }
  *vp8l = data;
  *vp8l_size = size;
  return WLL_OK;
}

wll_status wll_get_info(const uint8_t* data, size_t size, int* width,
                        int* height) {
  const uint8_t* vp8l;
  size_t vp8l_size;
  int w, h;
  const wll_status st = parse_header(data, size, &vp8l, &vp8l_size, &w, &h);
  if (st != WLL_OK) return st;
  if (width != NULL) *width = w;
  if (height != NULL) *height = h;
  return WLL_OK;
}

/* ------------------------------------------------------------------------ */

/* Frees the work arrays if they and the cached blocks together exceed the
 * cache limit. */
static void trim(wll_decoder* dec) {
  const size_t kept = dec->tables.cap * sizeof(hentry) +
                      dec->scratch.cap * sizeof(hentry) +
                      dec->groups_cap * sizeof(group) +
                      dec->pairs_cap * sizeof(uint32_t) +
                      dec->rows_cap * sizeof(uint32_t);
  if (kept + dec->mem.cached <= dec->mem.limit) return;
  free(dec->tables.entries);
  free(dec->scratch.entries);
  free(dec->groups);
  free(dec->pairs);
  free(dec->rows);
  dec->tables = (htables){NULL, 0, 0};
  dec->scratch = (htables){NULL, 0, 0};
  dec->groups = NULL;
  dec->pairs = dec->rows = NULL;
  dec->groups_cap = dec->pairs_cap = dec->rows_cap = 0;
}

static wll_status decode(wll_decoder* dec, const uint8_t* data, size_t size,
                         uint8_t* rgba, int rgba_width, int rgba_height,
                         size_t stride) {
  const uint8_t* vp8l;
  size_t vp8l_size;
  state st;
  memset(&st, 0, sizeof(st));
  wll_status status = parse_header(data, size, &vp8l, &vp8l_size, &st.width,
                                   &st.height);
  if (status != WLL_OK) return status;
  /* `rgba` holds rgba_width x rgba_height pixels. */
  if (rgba == NULL || st.width != rgba_width || st.height != rgba_height ||
      stride < (size_t)st.width * 4) {
    return WLL_ERR_INVALID_ARG;
  }
  const int width = st.width, height = st.height;
  st.dec = dec;
  st.rgba = rgba;
  st.stride = stride;
  st.br = (bitreader){.data = vp8l, .size = vp8l_size, .pos = 5};
  dec->tables.size = 0;
  uint32_t* used = NULL;   /* tiles per group */
  uint64_t* pixels = NULL; /* pixels per group */

  /* Transforms, each type at most once. Color indexing narrows the image
   * (several indices per pixel) for the transforms after it. */
  unsigned seen = 0;
  int xsize = width;
  while (br_read(&st.br, 1)) {
    const int type = (int)br_read(&st.br, 2);
    if (seen & (1u << type)) {
      status = WLL_ERR_INVALID_DATA;
      goto done;
    }
    seen |= 1u << type;
    transform* t = &st.transforms[st.num_transforms++];
    t->type = type;
    t->width = xsize;
    if (type == COLOR_INDEXING_TRANSFORM) {
      const int num_colors = (int)br_read(&st.br, 8) + 1;
      t->bits = num_colors > 16 ? 0 : num_colors > 4 ? 1 : num_colors > 2 ? 2 : 3;
      uint32_t* colors;
      status = decode_sub_image(&st, num_colors, 1, &colors);
      if (status != WLL_OK) goto done;
      /* Delta-coded; indices past the colors give 0, as in libwebp. */
      t->data = (uint32_t*)calloc(256, sizeof(uint32_t));
      if (t->data == NULL) {
        free(colors);
        status = WLL_ERR_OUT_OF_MEMORY;
        goto done;
      }
      t->data[0] = colors[0];
      for (int i = 1; i < num_colors; ++i) {
        t->data[i] = add_pixels(colors[i], t->data[i - 1]);
      }
      free(colors);
      xsize = (xsize + (1 << t->bits) - 1) >> t->bits;
    } else if (type != SUBTRACT_GREEN_TRANSFORM) {
      t->bits = (int)br_read(&st.br, 3) + 2;
      t->xsize = (xsize + (1 << t->bits) - 1) >> t->bits;
      status = decode_sub_image(&st, t->xsize,
                                (height + (1 << t->bits) - 1) >> t->bits,
                                &t->data);
      if (status != WLL_OK) goto done;
    }
  }
  st.xsize = xsize;
  const int cache_bits = read_cache_bits(&st.br);
  if (cache_bits < 0) {
    status = WLL_ERR_INVALID_DATA;
    goto done;
  }

  /* Meta prefix codes. */
  int num_groups = 1, num_used = 1;
  if (br_read(&st.br, 1)) {
    st.meta_bits = (int)br_read(&st.br, 3) + 2;
    st.meta_xsize = (xsize + (1 << st.meta_bits) - 1) >> st.meta_bits;
    const int meta_ysize = (height + (1 << st.meta_bits) - 1) >> st.meta_bits;
    status = decode_sub_image(&st, st.meta_xsize, meta_ysize, &st.meta);
    if (status != WLL_OK) goto done;
    const size_t tiles = (size_t)st.meta_xsize * (size_t)meta_ysize;
    for (size_t i = 0; i < tiles; ++i) {
      st.meta[i] = (st.meta[i] >> 8) & 0xffff;
      if ((int)st.meta[i] >= num_groups) num_groups = (int)st.meta[i] + 1;
    }
    used = (uint32_t*)calloc((size_t)num_groups, sizeof(*used));
    if (used == NULL) {
      status = WLL_ERR_OUT_OF_MEMORY;
      goto done;
    }
    num_used = 0;
    for (size_t i = 0; i < tiles; ++i) num_used += used[st.meta[i]]++ == 0;
    if (num_groups == 1) {
      free(st.meta);
      st.meta = NULL;
    }
  }
  {
    /* With many groups (which alternate along each row) smaller tables
     * keep more of them in the CPU caches. */
    const int many = num_used > 4;
    const int r = read_groups(&st, num_groups, used, many ? 8 : 10,
                              cache_bits);
    if (r <= 0) {
      status = r < 0 ? WLL_ERR_OUT_OF_MEMORY : WLL_ERR_INVALID_DATA;
      goto done;
    }
    /* Pixels per group (tiles at the edges count fully). */
    pixels = (uint64_t*)malloc((size_t)num_groups * sizeof(*pixels));
    if (pixels == NULL) {
      status = WLL_ERR_OUT_OF_MEMORY;
      goto done;
    }
    for (int i = 0; i < num_groups; ++i) {
      pixels[i] = used != NULL ? (uint64_t)used[i] << (2 * st.meta_bits)
                               : (uint64_t)xsize * (uint64_t)height;
    }
    if (!finish_groups(dec, num_groups, pixels,
                       many ? MANY_PAIR_BITS : PAIR_BITS)) {
      status = WLL_ERR_OUT_OF_MEMORY;
      goto done;
    }
  }

  /* Pixels. */
  {
    const size_t n = (size_t)xsize * (size_t)height;
    const size_t rows = 4 * (size_t)width;
    st.pixels = (uint32_t*)wll_mem_alloc(&dec->mem, n * sizeof(uint32_t));
    if (rows > dec->rows_cap) {
      free(dec->rows);
      dec->rows = (uint32_t*)malloc(rows * sizeof(uint32_t));
      dec->rows_cap = dec->rows != NULL ? rows : 0;
    }
    if (st.pixels == NULL || dec->rows == NULL) {
      status = WLL_ERR_OUT_OF_MEMORY;
      goto done;
    }
    st.tmp = dec->rows;
    st.tmp2 = dec->rows + width;
    st.pred_rows = dec->rows + 2 * (size_t)width;
    cache_start(&st, cache_bits);
    if (!decode_pixels(&st, st.pixels, xsize, height, 1)) {
      status = WLL_ERR_INVALID_DATA;
      goto done;
    }
    process_rows(&st, height);
  }
  status = WLL_OK;

done:
  wll_mem_free(&dec->mem, st.pixels);
  free(pixels);
  free(used);
  free(st.meta);
  for (int i = 0; i < st.num_transforms; ++i) free(st.transforms[i].data);
  trim(dec);
  return status;
}

wll_decoder* wll_decoder_new(void) {
  wll_decoder* dec = (wll_decoder*)calloc(1, sizeof(wll_decoder));
  if (dec != NULL) wll_mem_init(&dec->mem, WLL_DEFAULT_CACHE_LIMIT);
  return dec;
}

wll_status wll_decoder_set_cache_limit(wll_decoder* dec, size_t bytes) {
  if (dec == NULL) return WLL_ERR_INVALID_ARG;
  wll_mem_set_limit(&dec->mem, bytes);
  trim(dec);
  return WLL_OK;
}

void wll_decoder_free(wll_decoder* dec) {
  if (dec == NULL) return;
  wll_mem_release(&dec->mem);
  free(dec->tables.entries);
  free(dec->scratch.entries);
  free(dec->groups);
  free(dec->pairs);
  free(dec->rows);
  free(dec);
}

wll_status wll_decoder_decode(wll_decoder* dec, const uint8_t* data,
                              size_t size, uint8_t* rgba, int width,
                              int height, size_t stride) {
  if (dec == NULL) return WLL_ERR_INVALID_ARG;
#ifdef WLL_TIMING
  /* Everything but the row processing counts as entropy decoding. */
  const double rows = wll_timing[WLL_T_D_ROWS];
  WLL_TIMER_START(timer);
  const wll_status st =
      decode(dec, data, size, rgba, width, height, stride);
  WLL_TIMER_LAP(timer, WLL_T_D_ENTROPY);
  wll_timing[WLL_T_D_ENTROPY] -= wll_timing[WLL_T_D_ROWS] - rows;
  return st;
#else
  return decode(dec, data, size, rgba, width, height, stride);
#endif
}

wll_status wll_decode_rgba(const uint8_t* data, size_t size, uint8_t** rgba,
                           int* width, int* height) {
  int w, h;
  if (rgba == NULL) return WLL_ERR_INVALID_ARG;
  *rgba = NULL;
  wll_status status = wll_get_info(data, size, &w, &h);
  if (status != WLL_OK) return status;
  uint8_t* out = (uint8_t*)malloc((size_t)w * (size_t)h * 4);
  wll_decoder* dec = wll_decoder_new();
  if (out == NULL || dec == NULL) {
    free(out);
    wll_decoder_free(dec);
    return WLL_ERR_OUT_OF_MEMORY;
  }
  status = decode(dec, data, size, out, w, h, (size_t)w * 4);
  wll_decoder_free(dec);
  if (status != WLL_OK) {
    free(out);
    return status;
  }
  *rgba = out;
  if (width != NULL) *width = w;
  if (height != NULL) *height = h;
  return WLL_OK;
}
