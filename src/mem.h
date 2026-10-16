/* Copyright 2026 Anlatan Inc.
 * SPDX-License-Identifier: Apache-2.0 */

/* Allocation with a cache of large blocks.
 *
 * Large buffers freed through a wll_mem are kept and handed out again by
 * later allocations, within one encode and across encodes. This avoids the
 * page faults and kernel zeroing that fresh large allocations cost each
 * time. At most `limit` bytes are kept; blocks freed beyond that go back to
 * the heap. All functions accept m == NULL, which means plain malloc/free.
 */
#ifndef WLL_MEM_H
#define WLL_MEM_H

#include <stddef.h>

#define WLL_MEM_CACHED_BLOCKS 16

typedef struct {
  void* base;
  size_t cap;
} wll_mem_block;

typedef struct {
  wll_mem_block blocks[WLL_MEM_CACHED_BLOCKS];
  int num_blocks;
  size_t cached; /* total capacity of the blocks */
  size_t limit;  /* most bytes to keep */
} wll_mem;

/* An empty cache keeping at most `limit` bytes. */
void wll_mem_init(wll_mem* m, size_t limit);
void* wll_mem_alloc(wll_mem* m, size_t size);
/* Zero-initialized; returns NULL on overflow of n * size. */
void* wll_mem_calloc(wll_mem* m, size_t n, size_t size);
/* Accepts NULL. `p` must come from the same wll_mem (or both NULL). */
void wll_mem_free(wll_mem* m, void* p);
/* Frees all cached blocks. */
void wll_mem_release(wll_mem* m);
/* Changes the limit, freeing blocks beyond it. */
void wll_mem_set_limit(wll_mem* m, size_t limit);

#endif /* WLL_MEM_H */
