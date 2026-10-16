#include "mem.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Blocks at least this large are cached; smaller ones come cheaply from the
 * heap anyway. */
#define CACHE_MIN_SIZE (64 * 1024)
/* Header before each block holding its capacity; keeps 16-byte alignment. */
#define HEADER 16

static size_t* header_of(void* p) { return (size_t*)((uint8_t*)p - HEADER); }

void wll_mem_init(wll_mem* m, size_t limit) {
  m->num_blocks = 0;
  m->cached = 0;
  m->limit = limit;
}

static int smallest_block(const wll_mem* m) {
  int smallest = 0;
  for (int i = 1; i < m->num_blocks; ++i) {
    if (m->blocks[i].cap < m->blocks[smallest].cap) smallest = i;
  }
  return smallest;
}

static void drop_block(wll_mem* m, int i) {
  free(m->blocks[i].base);
  m->cached -= m->blocks[i].cap;
  m->blocks[i] = m->blocks[--m->num_blocks];
}

void* wll_mem_alloc(wll_mem* m, size_t size) {
  if (m != NULL && size >= CACHE_MIN_SIZE) {
    /* Smallest cached block that fits. */
    int best = -1;
    for (int i = 0; i < m->num_blocks; ++i) {
      if (m->blocks[i].cap >= size &&
          (best < 0 || m->blocks[i].cap < m->blocks[best].cap)) {
        best = i;
      }
    }
    if (best >= 0) {
      void* base = m->blocks[best].base;
      m->cached -= m->blocks[best].cap;
      m->blocks[best] = m->blocks[--m->num_blocks];
      return (uint8_t*)base + HEADER;
    }
  }
  if (size > SIZE_MAX - HEADER) return NULL;
  void* base = malloc(size + HEADER);
  if (base == NULL) return NULL;
  *(size_t*)base = size;
  return (uint8_t*)base + HEADER;
}

void* wll_mem_calloc(wll_mem* m, size_t n, size_t size) {
  if (size != 0 && n > SIZE_MAX / size) return NULL;
  void* p = wll_mem_alloc(m, n * size);
  if (p != NULL) memset(p, 0, n * size);
  return p;
}

void wll_mem_free(wll_mem* m, void* p) {
  if (p == NULL) return;
  void* base = header_of(p);
  const size_t cap = *(size_t*)base;
  if (m == NULL || cap < CACHE_MIN_SIZE || cap > m->limit) {
    free(base);
    return;
  }
  while (m->num_blocks == WLL_MEM_CACHED_BLOCKS ||
         cap > m->limit - m->cached) {
    /* Make room by dropping the smallest block, unless this one is smaller. */
    const int smallest = smallest_block(m);
    if (m->blocks[smallest].cap >= cap) {
      free(base);
      return;
    }
    drop_block(m, smallest);
  }
  m->blocks[m->num_blocks].base = base;
  m->blocks[m->num_blocks].cap = cap;
  ++m->num_blocks;
  m->cached += cap;
}

void wll_mem_release(wll_mem* m) {
  for (int i = 0; i < m->num_blocks; ++i) free(m->blocks[i].base);
  m->num_blocks = 0;
  m->cached = 0;
}

void wll_mem_set_limit(wll_mem* m, size_t limit) {
  m->limit = limit;
  while (m->cached > limit) drop_block(m, smallest_block(m));
}
