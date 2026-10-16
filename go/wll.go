// Package wll is the Go binding of wll, a lossless WebP (VP8L) encoder that
// keeps RGBA exactly, including RGB under alpha=0. The C sources in ../src
// are compiled into the package through the cgo_*.c files.
package wll

/*
#cgo CFLAGS: -O3 -std=c11
#cgo linux LDFLAGS: -lm -pthread
#include "../src/wll.h"
*/
import "C"

import (
	"bytes"
	"fmt"
	"math"
	"unsafe"
)

const MaxDimension = C.WLL_MAX_DIMENSION

type Level int

const (
	// LevelFast is about 25x faster than LevelDefault, at about 17% larger files.
	LevelFast Level = C.WLL_LEVEL_FAST
	// LevelDefault aims at `cwebp -lossless -exact -z 2` file sizes.
	LevelDefault Level = C.WLL_LEVEL_DEFAULT
)

// Encode encodes non-premultiplied RGBA pixels (rows stride bytes apart) as a
// complete WebP file. Each call creates and frees its own C encoder, as
// libwebp does: work buffers are reused within the call only, so no memory
// stays allocated between calls. It is safe to call from many goroutines.
func Encode(pix []byte, width, height, stride int, level Level) ([]byte, error) {
	if level != LevelFast && level != LevelDefault {
		return nil, fmt.Errorf("wll: unknown level %d", level)
	}
	if width <= 0 || height <= 0 || width > MaxDimension || height > MaxDimension {
		return nil, fmt.Errorf("wll: invalid dimensions %dx%d", width, height)
	}
	if stride < width*4 {
		return nil, fmt.Errorf("wll: stride %d below width*4 (%d)", stride, width*4)
	}
	// The C side reads (height-1)*stride + width*4 bytes; checking the
	// multiply first keeps a huge stride from wrapping need below len(pix).
	if height > 1 && stride > (math.MaxInt-width*4)/(height-1) {
		return nil, fmt.Errorf("wll: stride %d too large for height %d", stride, height)
	}
	if need := (height-1)*stride + width*4; len(pix) < need {
		return nil, fmt.Errorf("wll: pixel buffer has %d bytes, %dx%d with stride %d needs %d", len(pix), width, height, stride, need)
	}

	enc := C.wll_encoder_new()
	if enc == nil {
		return nil, fmt.Errorf("wll: out of memory")
	}
	defer C.wll_encoder_free(enc)
	if st := C.wll_encoder_set_level(enc, C.int(level)); st != C.WLL_OK {
		return nil, statusError(st)
	}

	var out *C.uint8_t
	var outSize C.size_t
	st := C.wll_encoder_encode(enc, (*C.uint8_t)(unsafe.Pointer(&pix[0])), C.int(width), C.int(height), C.size_t(stride), &out, &outSize)
	if st != C.WLL_OK {
		return nil, statusError(st)
	}
	defer C.wll_free(unsafe.Pointer(out))
	// Copied without C.GoBytes, whose C.int length would wrap above 2 GiB.
	return bytes.Clone(unsafe.Slice((*byte)(unsafe.Pointer(out)), outSize)), nil
}

func statusError(st C.wll_status) error {
	switch st {
	case C.WLL_ERR_INVALID_ARG:
		return fmt.Errorf("wll: invalid argument")
	case C.WLL_ERR_OUT_OF_MEMORY:
		return fmt.Errorf("wll: out of memory")
	default:
		return fmt.Errorf("wll: status %d", int(st))
	}
}
