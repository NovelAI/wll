/* Minimal reader for RGBA PAM files (as written by `dwebp -pam`). */
#ifndef WLL_TOOLS_PAM_H
#define WLL_TOOLS_PAM_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Returns a malloc'd RGBA buffer (stride 4*width), or NULL on error. */
static uint8_t* pam_read(const char* path, int* width, int* height) {
  FILE* f = fopen(path, "rb");
  if (f == NULL) return NULL;
  char line[256];
  int w = 0, h = 0, depth = 0, maxval = 0, ok = 0;
  uint8_t* px = NULL;
  size_t n;
  if (fgets(line, sizeof(line), f) == NULL || strncmp(line, "P7", 2) != 0) {
    goto done;
  }
  while (fgets(line, sizeof(line), f) != NULL) {
    if (sscanf(line, "WIDTH %d", &w) == 1) continue;
    if (sscanf(line, "HEIGHT %d", &h) == 1) continue;
    if (sscanf(line, "DEPTH %d", &depth) == 1) continue;
    if (sscanf(line, "MAXVAL %d", &maxval) == 1) continue;
    if (strncmp(line, "ENDHDR", 6) == 0) {
      ok = 1;
      break;
    }
  }
  if (!ok || w < 1 || h < 1 || w > 65535 || h > 65535 || depth != 4 ||
      maxval != 255) {
    ok = 0;
    goto done;
  }
  n = (size_t)w * (size_t)h * 4;
  px = (uint8_t*)malloc(n);
  if (px == NULL || fread(px, 1, n, f) != n) {
    free(px);
    px = NULL;
    goto done;
  }
  *width = w;
  *height = h;
done:
  fclose(f);
  return px;
}

#endif /* WLL_TOOLS_PAM_H */
