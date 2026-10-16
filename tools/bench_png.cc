// bench_png: compare wll against PNG encoders (libpng, fpnge, fpng) on RGBA
// PAM files. Every encoder runs in-process and single-threaded on the same
// pixels; the table lists output size and encoding speed.
//
// Build: see the bench-png target in the Makefile (needs the libpng, fpnge
// and fpng sources).
#include <png.h>
#include <time.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "fpng.h"
#include "fpnge.h"
extern "C" {
#include "pam.h"
#include "wll.h"
}

static double now() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

struct Image {
  std::vector<uint8_t> rgba;
  int w, h;
};

// Returns the encoded size, or 0 on failure.
using Encoder = std::function<size_t(const Image&)>;

static void png_write_vec(png_structp png, png_bytep data, png_size_t len) {
  auto* out = static_cast<std::vector<uint8_t>*>(png_get_io_ptr(png));
  out->insert(out->end(), data, data + len);
}

static size_t encode_libpng(const Image& im, int level) {
  std::vector<uint8_t> out;
  png_structp png =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  png_infop info = png_create_info_struct(png);
  if (setjmp(png_jmpbuf(png))) {
    png_destroy_write_struct(&png, &info);
    return 0;
  }
  png_set_write_fn(png, &out, png_write_vec, nullptr);
  png_set_IHDR(png, info, im.w, im.h, 8, PNG_COLOR_TYPE_RGBA,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);
  png_set_compression_level(png, level);
  png_write_info(png, info);
  for (int y = 0; y < im.h; ++y) {
    png_write_row(png, im.rgba.data() + (size_t)y * im.w * 4);
  }
  png_write_end(png, info);
  png_destroy_write_struct(&png, &info);
  return out.size();
}

static size_t encode_fpnge(const Image& im, int level) {
  FPNGEOptions options;
  FPNGEFillOptions(&options, level, FPNGE_CICP_NONE);
  std::vector<uint8_t> out(FPNGEOutputAllocSize(1, 4, im.w, im.h));
  return FPNGEEncode(1, 4, im.rgba.data(), im.w, (size_t)im.w * 4, im.h,
                     out.data(), &options);
}

static size_t encode_fpng(const Image& im, uint32_t flags) {
  std::vector<uint8_t> out;
  if (!fpng::fpng_encode_image_to_memory(im.rgba.data(), im.w, im.h, 4, out,
                                         flags)) {
    return 0;
  }
  return out.size();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s images.pam...\n", argv[0]);
    return 2;
  }
  fpng::fpng_init();
  wll_encoder* enc = wll_encoder_new();
  wll_encoder* enc_fast = wll_encoder_new();
  wll_encoder_set_level(enc_fast, WLL_LEVEL_FAST);
  struct Entry {
    const char* name;
    Encoder fn;
    double bytes = 0, seconds = 0;
  };
  std::vector<Entry> entries = {
      {"wll",
       [&](const Image& im) -> size_t {
         uint8_t* out;
         size_t size;
         if (wll_encoder_encode(enc, im.rgba.data(), im.w, im.h,
                                (size_t)im.w * 4, &out, &size) != WLL_OK) {
           return 0;
         }
         wll_free(out);
         return size;
       }},
      {"wll fast",
       [&](const Image& im) -> size_t {
         uint8_t* out;
         size_t size;
         if (wll_encoder_encode(enc_fast, im.rgba.data(), im.w, im.h,
                                (size_t)im.w * 4, &out, &size) != WLL_OK) {
           return 0;
         }
         wll_free(out);
         return size;
       }},
      {"libpng -1", [](const Image& im) { return encode_libpng(im, 1); }},
      {"libpng -6 (default)", [](const Image& im) { return encode_libpng(im, 6); }},
      {"libpng -9", [](const Image& im) { return encode_libpng(im, 9); }},
      {"fpnge 1", [](const Image& im) { return encode_fpnge(im, 1); }},
      {"fpnge 4 (default)", [](const Image& im) { return encode_fpnge(im, 4); }},
      {"fpnge 5", [](const Image& im) { return encode_fpnge(im, 5); }},
      {"fpng", [](const Image& im) { return encode_fpng(im, 0); }},
      {"fpng slower",
       [](const Image& im) { return encode_fpng(im, fpng::FPNG_ENCODE_SLOWER); }},
  };
  double pixels = 0;
  for (int i = 1; i < argc; ++i) {
    Image im;
    uint8_t* p = pam_read(argv[i], &im.w, &im.h);
    if (p == nullptr) {
      fprintf(stderr, "cannot read %s\n", argv[i]);
      return 1;
    }
    im.rgba.assign(p, p + (size_t)im.w * im.h * 4);
    free(p);
    pixels += (double)im.w * im.h;
    for (Entry& e : entries) {
      const double t0 = now();
      const size_t size = e.fn(im);
      e.seconds += now() - t0;
      if (size == 0) {
        fprintf(stderr, "%s failed on %s\n", e.name, argv[i]);
        return 1;
      }
      e.bytes += (double)size;
    }
  }
  const double wll_bytes = entries[0].bytes;
  printf("%d images, %.1f MP\n", argc - 1, pixels / 1e6);
  printf("| encoder | bpp | size vs wll | MP/s |\n|---|---|---|---|\n");
  for (const Entry& e : entries) {
    printf("| %s | %.3f | %.1f%% | %.2f |\n", e.name, e.bytes * 8 / pixels,
           100. * e.bytes / wll_bytes, pixels / e.seconds / 1e6);
  }
  wll_encoder_free(enc_fast);
  wll_encoder_free(enc);
  return 0;
}
