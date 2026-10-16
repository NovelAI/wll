#include "bitwriter.h"

#include <string.h>

int wll_bw_init(wll_bw* bw, size_t initial_cap, wll_mem* mem) {
  bw->mem = mem;
  bw->pos = 0;
  bw->acc = 0;
  bw->nbits = 0;
  bw->error = 0;
  bw->cap = initial_cap < 64 ? 64 : initial_cap;
  bw->buf = (uint8_t*)wll_mem_alloc(mem, bw->cap);
  if (bw->buf == NULL) {
    bw->cap = 0;
    bw->error = 1;
    return 0;
  }
  return 1;
}

void wll_bw_free(wll_bw* bw) {
  wll_mem_free(bw->mem, bw->buf);
  bw->buf = NULL;
  bw->cap = bw->pos = 0;
}

int wll_bw_reserve(wll_bw* bw, size_t extra) {
  if (bw->error) return 0;
  if (bw->cap - bw->pos >= extra) return 1;
  size_t need = bw->pos + extra;
  if (need < bw->pos) goto fail; /* overflow */
  size_t cap = bw->cap;
  while (cap < need) {
    if (cap > SIZE_MAX / 2) goto fail;
    cap *= 2;
  }
  uint8_t* buf = (uint8_t*)wll_mem_alloc(bw->mem, cap);
  if (buf == NULL) goto fail;
  memcpy(buf, bw->buf, bw->pos);
  wll_mem_free(bw->mem, bw->buf);
  bw->buf = buf;
  bw->cap = cap;
  return 1;
fail:
  bw->error = 1;
  return 0;
}

void wll_bw_flush(wll_bw* bw) {
  int nbytes = (bw->nbits + 7) >> 3;
  if (!wll_bw_reserve(bw, (size_t)nbytes)) return;
  for (int i = 0; i < nbytes; ++i) {
    bw->buf[bw->pos++] = (uint8_t)bw->acc;
    bw->acc >>= 8;
  }
  bw->acc = 0;
  bw->nbits = 0;
}
