// Copyright 2026 Anlatan Inc.
// SPDX-License-Identifier: Apache-2.0

// bench_png_decode: decoding speed of PNG decoders (libpng, libspng, Wuffs,
// fpng) and WebP decoders (libwebp, wll) on the same RGBA PAM images.
//
// Each image is encoded once per file format; then every decoder that can
// read the format decodes it into a preallocated RGBA buffer, single-
// threaded. The fastest of `rounds` runs per image counts, and every output
// is checked against the original pixels. The PNG decoders verify
// checksums (zlib and chunk CRCs) as by default; WebP has none.
//
// Build: see the bench-png-decode target in the Makefile.
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
#include "spng.h"
#include "webp/decode.h"
#include "wuffs-v0.3.c"
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

using File = std::vector<uint8_t>;

// ---------------------------------------------------------------------------
// Encoders

static void png_write_vec(png_structp png, png_bytep data, png_size_t len) {
  auto* out = static_cast<File*>(png_get_io_ptr(png));
  out->insert(out->end(), data, data + len);
}

static File encode_libpng(const Image& im) {
  File out;
  png_structp png =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  png_infop info = png_create_info_struct(png);
  if (setjmp(png_jmpbuf(png))) {
    png_destroy_write_struct(&png, &info);
    return {};
  }
  png_set_write_fn(png, &out, png_write_vec, nullptr);
  png_set_IHDR(png, info, im.w, im.h, 8, PNG_COLOR_TYPE_RGBA,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png, info);
  for (int y = 0; y < im.h; ++y) {
    png_write_row(png, im.rgba.data() + (size_t)y * im.w * 4);
  }
  png_write_end(png, info);
  png_destroy_write_struct(&png, &info);
  return out;
}

static File encode_fpnge(const Image& im) {
  FPNGEOptions options;
  FPNGEFillOptions(&options, FPNGE_COMPRESS_LEVEL_DEFAULT, FPNGE_CICP_NONE);
  File out(FPNGEOutputAllocSize(1, 4, im.w, im.h));
  out.resize(FPNGEEncode(1, 4, im.rgba.data(), im.w, (size_t)im.w * 4, im.h,
                         out.data(), &options));
  return out;
}

static File encode_fpng(const Image& im) {
  File out;
  if (!fpng::fpng_encode_image_to_memory(im.rgba.data(), im.w, im.h, 4,
                                         out)) {
    return {};
  }
  return out;
}

static File encode_wll(const Image& im, int level) {
  wll_encoder* enc = wll_encoder_new();
  wll_encoder_set_level(enc, level);
  uint8_t* out;
  size_t size;
  File file;
  if (wll_encoder_encode(enc, im.rgba.data(), im.w, im.h, (size_t)im.w * 4,
                         &out, &size) == WLL_OK) {
    file.assign(out, out + size);
    wll_free(out);
  }
  wll_encoder_free(enc);
  return file;
}

// ---------------------------------------------------------------------------
// Decoders: decode `file` (w x h) into `out` (w * 4 bytes per row); return
// false on failure.

using Decoder = std::function<bool(const File&, int, int, uint8_t*)>;

struct MemReader {
  const File* file;
  size_t pos;
};

static void png_read_mem(png_structp png, png_bytep data, png_size_t len) {
  auto* r = static_cast<MemReader*>(png_get_io_ptr(png));
  if (len > r->file->size() - r->pos) png_error(png, "read past end");
  memcpy(data, r->file->data() + r->pos, len);
  r->pos += len;
}

static bool decode_libpng(const File& file, int w, int h, uint8_t* out) {
  png_structp png =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  png_infop info = png_create_info_struct(png);
  std::vector<png_bytep> rows(h);
  MemReader reader = {&file, 0};
  if (setjmp(png_jmpbuf(png))) {
    png_destroy_read_struct(&png, &info, nullptr);
    return false;
  }
  png_set_read_fn(png, &reader, png_read_mem);
  png_read_info(png, info);
  if ((int)png_get_image_width(png, info) != w ||
      (int)png_get_image_height(png, info) != h ||
      png_get_color_type(png, info) != PNG_COLOR_TYPE_RGBA ||
      png_get_bit_depth(png, info) != 8) {
    png_destroy_read_struct(&png, &info, nullptr);
    return false;
  }
  for (int y = 0; y < h; ++y) rows[y] = out + (size_t)y * w * 4;
  png_read_image(png, rows.data());
  png_read_end(png, nullptr);
  png_destroy_read_struct(&png, &info, nullptr);
  return true;
}

static bool decode_spng(const File& file, int w, int h, uint8_t* out) {
  spng_ctx* ctx = spng_ctx_new(0);
  bool ok = ctx != nullptr &&
            spng_set_png_buffer(ctx, file.data(), file.size()) == 0 &&
            spng_decode_image(ctx, out, (size_t)w * h * 4, SPNG_FMT_RGBA8,
                              0) == 0;
  spng_ctx_free(ctx);
  return ok;
}

static bool decode_wuffs(const File& file, int w, int h, uint8_t* out) {
  static std::vector<uint8_t> work;
  wuffs_png__decoder* dec = wuffs_png__decoder__alloc();
  if (dec == nullptr) return false;
  wuffs_base__image_config ic;
  wuffs_base__io_buffer src = wuffs_base__ptr_u8__reader(
      const_cast<uint8_t*>(file.data()), file.size(), true);
  bool ok = false;
  wuffs_base__status st =
      wuffs_png__decoder__decode_image_config(dec, &ic, &src);
  if (wuffs_base__status__is_ok(&st) &&
      (int)wuffs_base__pixel_config__width(&ic.pixcfg) == w &&
      (int)wuffs_base__pixel_config__height(&ic.pixcfg) == h) {
    wuffs_base__pixel_config__set(&ic.pixcfg,
                                  WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL,
                                  WUFFS_BASE__PIXEL_SUBSAMPLING__NONE, w, h);
    const uint64_t work_len = wuffs_png__decoder__workbuf_len(dec).max_incl;
    if (work.size() < work_len) work.resize(work_len);
    wuffs_base__pixel_buffer pb;
    st = wuffs_base__pixel_buffer__set_from_slice(
        &pb, &ic.pixcfg, wuffs_base__make_slice_u8(out, (size_t)w * h * 4));
    if (wuffs_base__status__is_ok(&st)) {
      st = wuffs_png__decoder__decode_frame(
          dec, &pb, &src, WUFFS_BASE__PIXEL_BLEND__SRC,
          wuffs_base__make_slice_u8(work.data(), work_len), nullptr);
      ok = wuffs_base__status__is_ok(&st);
    }
  }
  free(dec);
  return ok;
}

static bool decode_fpng(const File& file, int w, int h, uint8_t* out) {
  static std::vector<uint8_t> pixels;
  uint32_t fw, fh, channels;
  if (fpng::fpng_decode_memory(file.data(), (uint32_t)file.size(), pixels, fw,
                               fh, channels, 4) != fpng::FPNG_DECODE_SUCCESS ||
      (int)fw != w || (int)fh != h) {
    return false;
  }
  // fpng decodes into its own vector; the copy is part of the cost.
  memcpy(out, pixels.data(), (size_t)w * h * 4);
  return true;
}

static bool decode_libwebp(const File& file, int w, int h, uint8_t* out) {
  return WebPDecodeRGBAInto(file.data(), file.size(), out, (size_t)w * h * 4,
                            w * 4) != nullptr;
}

static bool decode_wll(const File& file, int w, int h, uint8_t* out) {
  static wll_decoder* dec = wll_decoder_new();
  return wll_decoder_decode(dec, file.data(), file.size(), out, w, h,
                            (size_t)w * 4) == WLL_OK;
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  int rounds = 3, first = 1;
  if (argc > 2 && strcmp(argv[1], "-r") == 0) {
    rounds = atoi(argv[2]);
    first = 3;
  }
  if (first >= argc || rounds < 1) {
    fprintf(stderr, "usage: %s [-r rounds] images.pam...\n", argv[0]);
    return 2;
  }
  fpng::fpng_init();
  struct Format {
    const char* name;
    std::function<File(const Image&)> encode;
    std::vector<std::pair<const char*, Decoder>> decoders;
    double bytes = 0;
    std::vector<double> seconds;
  };
  const std::vector<std::pair<const char*, Decoder>> png_decoders = {
      {"libpng", decode_libpng},
      {"libspng", decode_spng},
      {"Wuffs", decode_wuffs}};
  std::vector<Format> formats = {
      {"PNG (libpng -6)", encode_libpng, png_decoders},
      {"PNG (fpnge 4)", encode_fpnge, png_decoders},
      {"PNG (fpng)", encode_fpng, png_decoders},
      {"WebP (wll)",
       [](const Image& im) { return encode_wll(im, WLL_LEVEL_DEFAULT); },
       {{"libwebp", decode_libwebp}, {"wll", decode_wll}}},
      {"WebP (wll fast)",
       [](const Image& im) { return encode_wll(im, WLL_LEVEL_FAST); },
       {{"libwebp", decode_libwebp}, {"wll", decode_wll}}},
  };
  formats[2].decoders.push_back({"fpng", decode_fpng});
  for (Format& f : formats) f.seconds.assign(f.decoders.size(), 0);

  double pixels = 0;
  std::vector<uint8_t> out;
  for (int i = first; i < argc; ++i) {
    Image im;
    uint8_t* p = pam_read(argv[i], &im.w, &im.h);
    if (p == nullptr) {
      fprintf(stderr, "cannot read %s\n", argv[i]);
      return 1;
    }
    im.rgba.assign(p, p + (size_t)im.w * im.h * 4);
    free(p);
    pixels += (double)im.w * im.h;
    out.resize(im.rgba.size());
    for (Format& f : formats) {
      const File file = f.encode(im);
      if (file.empty()) {
        fprintf(stderr, "%s failed on %s\n", f.name, argv[i]);
        return 1;
      }
      f.bytes += (double)file.size();
      for (size_t d = 0; d < f.decoders.size(); ++d) {
        double best = 1e30;
        for (int r = 0; r < rounds; ++r) {
          memset(out.data(), 0, out.size());
          const double t0 = now();
          const bool ok = f.decoders[d].second(file, im.w, im.h, out.data());
          const double t = now() - t0;
          if (!ok || out != im.rgba) {
            fprintf(stderr, "%s: %s %s on %s\n", f.name, f.decoders[d].first,
                    ok ? "decodes wrongly" : "fails", argv[i]);
            return 1;
          }
          if (t < best) best = t;
        }
        f.seconds[d] += best;
      }
    }
  }
  printf("%d images, %.1f MP\n", argc - first, pixels / 1e6);
  printf("| file | bpp | decoder | MP/s |\n|---|---|---|---|\n");
  for (const Format& f : formats) {
    for (size_t d = 0; d < f.decoders.size(); ++d) {
      char bpp[32] = "";
      if (d == 0) snprintf(bpp, sizeof(bpp), "%.3f", f.bytes * 8 / pixels);
      printf("| %s | %s | %s | %.2f |\n", d == 0 ? f.name : "", bpp,
             f.decoders[d].first, pixels / f.seconds[d] / 1e6);
    }
  }
  return 0;
}
