CC = cc
OPT = -O3 -g
CFLAGS = $(OPT) -std=c11 -Wall -Wextra -Wshadow -Isrc -pthread
LIBWEBP = third_party/libwebp
WEBP_CFLAGS = -I$(LIBWEBP)/src
WEBP_LIBS = $(LIBWEBP)/build/libwebp.a $(LIBWEBP)/build/libsharpyuv.a -lm

SRC = src/enc.c src/huffman.c src/bitwriter.c src/entropy.c src/transforms.c \
  src/backward_refs.c src/histogram.c src/mem.c src/fast.c src/dec.c
HDR = src/wll.h src/internal.h src/huffman.h src/bitwriter.h src/entropy.h \
  src/transforms.h src/backward_refs.h src/histogram.h src/mem.h \
  src/timing.h src/fast.h tools/pam.h

all: build/wllenc build/roundtrip

build/wllenc: tools/wllenc.c $(SRC) $(HDR) | build
	$(CC) $(CFLAGS) -o $@ tools/wllenc.c $(SRC) -lm

build/roundtrip: tools/roundtrip.c $(SRC) $(HDR) | build
	$(CC) $(CFLAGS) $(WEBP_CFLAGS) -o $@ tools/roundtrip.c $(SRC) $(WEBP_LIBS)

# Address/undefined behaviour sanitizer build of the round-trip test.
build/roundtrip-asan: tools/roundtrip.c $(SRC) $(HDR) | build
	$(CC) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
	  -fno-sanitize-recover=all -std=c11 -Wall -Wextra -Isrc -pthread $(WEBP_CFLAGS) \
	  -o $@ tools/roundtrip.c $(SRC) $(WEBP_LIBS)

# Round-trip test that also reports time per encoder stage.
build/roundtrip-timing: tools/roundtrip.c $(SRC) $(HDR) src/timing.h | build
	$(CC) $(CFLAGS) -DWLL_TIMING -D_POSIX_C_SOURCE=199309L $(WEBP_CFLAGS) \
	  -o $@ tools/roundtrip.c $(SRC) $(WEBP_LIBS)

# Decoding speed.
build/bench-decode: tools/bench_decode.c $(SRC) $(HDR) | build
	$(CC) $(CFLAGS) -D_POSIX_C_SOURCE=199309L $(WEBP_CFLAGS) -o $@ \
	  tools/bench_decode.c $(SRC) $(WEBP_LIBS)

build/bench-decode-timing: tools/bench_decode.c $(SRC) $(HDR) | build
	$(CC) $(CFLAGS) -DWLL_TIMING -D_POSIX_C_SOURCE=199309L $(WEBP_CFLAGS) \
	  -o $@ tools/bench_decode.c $(SRC) $(WEBP_LIBS)

# Peak memory of one encode or decode, against libwebp.
build/memuse: tools/memuse.c $(SRC) $(HDR) | build
	$(CC) $(CFLAGS) $(WEBP_CFLAGS) -o $@ tools/memuse.c $(SRC) $(WEBP_LIBS)

# Decoder fuzzing with libFuzzer (needs clang): build/fuzz-decode corpus_dir
build/fuzz-decode: tools/fuzz_decode.c $(SRC) $(HDR) | build
	clang -O1 -g -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all \
	  -Isrc $(WEBP_CFLAGS) -o $@ tools/fuzz_decode.c $(SRC) $(WEBP_LIBS)

# Encoder fuzzing: round trips of arbitrary pixels at both levels.
build/fuzz-encode: tools/fuzz_encode.c $(SRC) $(HDR) | build
	clang -O1 -g -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all \
	  -Isrc $(WEBP_CFLAGS) -o $@ tools/fuzz_encode.c $(SRC) $(WEBP_LIBS)

# Comparison against PNG encoders. PNGSRC holds libpng (built with CMake into
# $(PNGSRC)/libpng/build), fpnge and fpng checkouts; zlib comes from the
# system.
PNGSRC = third_party/png
build/bench-png: tools/bench_png.cc $(SRC) $(HDR) | build
	$(CC) $(CFLAGS) -c $(SRC)
	mv *.o build/
	$(CXX) -O3 -march=native -std=c++17 -Isrc -Itools \
	  -I$(PNGSRC)/libpng -I$(PNGSRC)/libpng/build -I$(PNGSRC)/fpnge \
	  -I$(PNGSRC)/fpng/src -o $@ tools/bench_png.cc \
	  $(PNGSRC)/fpnge/fpnge.cc $(PNGSRC)/fpng/src/fpng.cpp \
	  $(patsubst src/%.c,build/%.o,$(SRC)) \
	  $(PNGSRC)/libpng/build/libpng16.a -lz -lm -pthread

# Decoding speed against PNG decoders; PNGSRC also holds Wuffs (v0.3) and
# libspng checkouts.
WUFFS_MODULES = -DWUFFS_IMPLEMENTATION -DWUFFS_CONFIG__MODULES \
  -DWUFFS_CONFIG__MODULE__BASE -DWUFFS_CONFIG__MODULE__ADLER32 \
  -DWUFFS_CONFIG__MODULE__CRC32 -DWUFFS_CONFIG__MODULE__DEFLATE \
  -DWUFFS_CONFIG__MODULE__PNG -DWUFFS_CONFIG__MODULE__ZLIB
build/bench-png-decode: tools/bench_png_decode.cc $(SRC) $(HDR) | build
	$(CC) $(CFLAGS) -c $(SRC)
	mv *.o build/
	$(CC) -O3 -march=native -c $(WUFFS_MODULES) \
	  -o build/wuffs.o $(PNGSRC)/wuffs/release/c/wuffs-v0.3.c
	$(CC) -O3 -march=native -c -o build/spng.o $(PNGSRC)/spng/spng/spng.c
	$(CXX) -O3 -march=native -std=c++17 -Isrc -Itools $(WEBP_CFLAGS) \
	  -I$(PNGSRC)/libpng -I$(PNGSRC)/libpng/build -I$(PNGSRC)/fpnge \
	  -I$(PNGSRC)/fpng/src -I$(PNGSRC)/spng/spng \
	  -I$(PNGSRC)/wuffs/release/c -o $@ tools/bench_png_decode.cc \
	  $(PNGSRC)/fpnge/fpnge.cc $(PNGSRC)/fpng/src/fpng.cpp \
	  build/wuffs.o build/spng.o $(patsubst src/%.c,build/%.o,$(SRC)) \
	  $(PNGSRC)/libpng/build/libpng16.a $(WEBP_LIBS) -lz -lm -pthread

test: build/roundtrip build/roundtrip-asan
	build/roundtrip-asan
	build/roundtrip

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all test clean
