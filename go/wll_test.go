// Copyright 2026 Anlatan Inc.
// SPDX-License-Identifier: Apache-2.0

package wll

import (
	"bytes"
	"image"
	"math"
	"math/rand/v2"
	"sync"
	"testing"

	"golang.org/x/image/webp"
)

// randomRGBA returns pixels where half of them are fully transparent with
// random RGB, which encoders that are not exact would rewrite.
func randomRGBA(width, height, stride int, seed uint64) []byte {
	rng := rand.New(rand.NewPCG(seed, seed))
	pix := make([]byte, stride*height)
	for i := range pix {
		pix[i] = byte(rng.IntN(256))
	}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			if rng.IntN(2) == 0 {
				pix[y*stride+x*4+3] = 0
			}
		}
	}
	return pix
}

// Encodes from many goroutines at once, at both levels and with padded rows,
// and decodes with an independent decoder.
func TestEncode_RoundTripsExactly(t *testing.T) {
	const width, height = 67, 45
	var wg sync.WaitGroup
	for i := range 16 {
		wg.Go(func() {
			level := []Level{LevelFast, LevelDefault}[i%2]
			stride := width*4 + (i%3)*4
			pix := randomRGBA(width, height, stride, uint64(i))

			out, err := Encode(pix, width, height, stride, level)
			if err != nil {
				t.Errorf("level %d: %v", level, err)
				return
			}
			img, err := webp.Decode(bytes.NewReader(out))
			if err != nil {
				t.Errorf("level %d: decode: %v", level, err)
				return
			}
			nrgba, ok := img.(*image.NRGBA)
			if !ok {
				t.Errorf("lossless WebP decodes to NRGBA, got %T", img)
				return
			}
			if got := nrgba.Bounds(); got != image.Rect(0, 0, width, height) {
				t.Errorf("bounds %v", got)
				return
			}
			for y := range height {
				want := pix[y*stride : y*stride+width*4]
				got := nrgba.Pix[y*nrgba.Stride : y*nrgba.Stride+width*4]
				if !bytes.Equal(want, got) {
					t.Errorf("level %d: row %d differs", level, y)
					return
				}
			}
		})
	}
	wg.Wait()
}

func TestEncode_RejectsBadArguments(t *testing.T) {
	pix := make([]byte, 8*8*4)
	for _, tc := range []struct {
		name                  string
		pix                   []byte
		width, height, stride int
		level                 Level
	}{
		{"buffer too short", pix[:len(pix)-1], 8, 8, 32, LevelDefault},
		{"stride below width", pix, 8, 8, 28, LevelDefault},
		{"stride overflows need", pix, 8, 3, math.MaxInt/2 + 1, LevelDefault},
		{"zero width", pix, 0, 8, 32, LevelDefault},
		{"too large", pix, MaxDimension + 1, 1, 32, LevelDefault},
		{"unknown level", pix, 8, 8, 32, Level(7)},
	} {
		if _, err := Encode(tc.pix, tc.width, tc.height, tc.stride, tc.level); err == nil {
			t.Errorf("%s: expected an error", tc.name)
		}
	}
}
