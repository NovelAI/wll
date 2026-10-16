/* Copyright 2026 Anlatan Inc.
 * SPDX-License-Identifier: Apache-2.0 */

/* memuse: peak memory of one encode or decode, wll against libwebp.
 *
 * Usage: memuse MODE file
 *   enc-wll, enc-wll-fast, enc-libwebp (lossless -z 2, exact): file.pam
 *   dec-wll, dec-libwebp: file.webp
 * Prints the peak resident memory the operation adds to the process: the
 * baseline is taken after reading the input and, when decoding, after
 * allocating and touching the RGBA output. Run each mode in a new process. */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

#include "pam.h"
#include "webp/decode.h"
#include "webp/encode.h"
#include "wll.h"

static long peak_kb(void) {
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_maxrss;
}

/* Current resident memory (Linux). */
static long current_kb(void) {
  long pages = 0, resident = 0;
  FILE* f = fopen("/proc/self/statm", "r");
  if (f == NULL) return 0;
  if (fscanf(f, "%ld %ld", &pages, &resident) != 2) resident = 0;
  fclose(f);
  return resident * 4;
}

/* Memory a wll encoder or decoder keeps for reuse between calls. */
static long retained_kb = -1;

static uint8_t* read_file(const char* path, size_t* size) {
  FILE* f = fopen(path, "rb");
  if (f == NULL) return NULL;
  fseek(f, 0, SEEK_END);
  const long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t* data = n > 0 ? (uint8_t*)malloc((size_t)n) : NULL;
  if (data != NULL && fread(data, 1, (size_t)n, f) != (size_t)n) {
    free(data);
    data = NULL;
  }
  fclose(f);
  *size = (size_t)n;
  return data;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s MODE file\n", argv[0]);
    return 2;
  }
  const char* mode = argv[1];
  int w, h;
  long base;
  size_t out_size = 0;
  if (strncmp(mode, "enc-", 4) == 0) {
    uint8_t* rgba = pam_read(argv[2], &w, &h);
    if (rgba == NULL) return 1;
    base = peak_kb();
    if (strcmp(mode, "enc-libwebp") == 0) {
      WebPConfig config;
      WebPPicture pic;
      WebPMemoryWriter writer;
      if (!WebPConfigInit(&config) || !WebPConfigLosslessPreset(&config, 2) ||
          !WebPPictureInit(&pic)) {
        return 1;
      }
      config.exact = 1;
      pic.use_argb = 1;
      pic.width = w;
      pic.height = h;
      WebPMemoryWriterInit(&writer);
      pic.writer = WebPMemoryWrite;
      pic.custom_ptr = &writer;
      if (!WebPPictureImportRGBA(&pic, rgba, w * 4) ||
          !WebPEncode(&config, &pic)) {
        return 1;
      }
      out_size = writer.size;
      WebPPictureFree(&pic);
      WebPMemoryWriterClear(&writer);
    } else {
      wll_encoder* enc = wll_encoder_new();
      if (strcmp(mode, "enc-wll-fast") == 0) {
        wll_encoder_set_level(enc, WLL_LEVEL_FAST);
      }
      uint8_t* out;
      const long before = current_kb();
      if (wll_encoder_encode(enc, rgba, w, h, (size_t)w * 4, &out,
                             &out_size) != WLL_OK) {
        return 1;
      }
      wll_free(out);
      retained_kb = current_kb() - before;
      wll_encoder_free(enc);
    }
  } else {
    size_t size;
    uint8_t* data = read_file(argv[2], &size);
    if (data == NULL || !WebPGetInfo(data, size, &w, &h)) return 1;
    uint8_t* rgba = (uint8_t*)malloc((size_t)w * h * 4);
    /* Not zero: malloc + memset 0 can turn into calloc's untouched pages. */
    memset(rgba, 1, (size_t)w * h * 4);
    base = peak_kb();
    if (strcmp(mode, "dec-libwebp") == 0) {
      if (WebPDecodeRGBAInto(data, size, rgba, (size_t)w * h * 4, w * 4) ==
          NULL) {
        return 1;
      }
    } else {
      wll_decoder* dec = wll_decoder_new();
      const long before = current_kb();
      if (wll_decoder_decode(dec, data, size, rgba, w, h, (size_t)w * 4) !=
          WLL_OK) {
        return 1;
      }
      retained_kb = current_kb() - before;
      wll_decoder_free(dec);
    }
  }
  const long added = peak_kb() - base;
  printf("%-13s %dx%d: +%.1f MB (%.1f bytes/pixel)", mode, w, h,
         added / 1024.0, added * 1024.0 / ((double)w * h));
  if (retained_kb >= 0) {
    printf(", kept between calls: %.1f MB", retained_kb / 1024.0);
  }
  printf("\n");
  (void)out_size;
  return 0;
}
