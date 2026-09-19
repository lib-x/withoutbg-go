package withoutbg

import (
	"image"
	"image/color"
	_ "image/png"
	"os"
	"path/filepath"
	"testing"
)

// splitPanels cuts the three-panel sample image (input | cutout | matte) into
// its panels. The published samples are 3 x panelWidth wide.
func splitPanels(t *testing.T, path string) (input, matte *image.RGBA, w, h int) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("sample image not found: %v", err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	b := img.Bounds()
	pw := b.Dx() / 3
	if pw == 0 {
		t.Fatalf("sample image %dx%d is too small to split", b.Dx(), b.Dy())
	}
	rgba := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			rgba.Set(x, y, img.At(b.Min.X+x, b.Min.Y+y))
		}
	}
	crop := func(x0 int) *image.RGBA {
		out := image.NewRGBA(image.Rect(0, 0, pw, b.Dy()))
		for y := 0; y < b.Dy(); y++ {
			copy(out.Pix[y*out.Stride:y*out.Stride+pw*4], rgba.Pix[y*rgba.Stride+x0*4:])
		}
		return out
	}
	return crop(0), crop(2 * pw), pw, b.Dy()
}

// matteOf returns the R channel of a panel as an alpha image.
func matteOf(panel *image.RGBA) *image.Gray {
	b := panel.Bounds()
	out := image.NewGray(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			c := panel.RGBAAt(x, y)
			out.Pix[y*out.Stride+x] = c.R
		}
	}
	return out
}

// skipWithoutModel skips the test when the 455 MB model is not on disk. The
// ONNX Runtime shared library is resolved by ONNXRUNTIME_LIB_PATH or, when that
// is unset, by the pure-onnx bootstrap download.
func skipWithoutModel(t *testing.T) {
	if _, err := os.Stat(filepath.Join("testdata", "withoutbg-open-weights.onnx")); err != nil {
		t.Skipf("model not present: %v", err)
	}
}

// TestAlphaGolden compares the model's matte against the matte published with
// each sample image (the samples are three-panel strips: input | cutout |
// matte). The published panels are downscaled copies of the images the samples
// were produced from, so edge pixels cannot match exactly; the test therefore
// asserts aggregate agreement (mean absolute error and foreground IoU at 0.5)
// with enough headroom for the resampling difference, and logs the numbers.
func TestAlphaGolden(t *testing.T) {
	skipWithoutModel(t)
	r, err := New(Config{ModelPath: filepath.Join("testdata", "withoutbg-open-weights.onnx")})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	t.Logf("contract: %s", r.Sidecar())

	for _, name := range []string{"example1.png", "example2.png", "example3.png"} {
		t.Run(name, func(t *testing.T) {
			input, refPanel, w, h := splitPanels(t, filepath.Join("testdata", name))
			ref := matteOf(refPanel)
			got, err := r.Alpha(input)
			if err != nil {
				t.Fatal(err)
			}
			if got.Bounds().Dx() != w || got.Bounds().Dy() != h {
				t.Fatalf("alpha size %v, want %dx%d", got.Bounds(), w, h)
			}
			var inter, union, diff, n int
			for y := range h {
				for x := range w {
					g := got.Pix[y*got.Stride+x]
					e := ref.Pix[y*ref.Stride+x]
					if g > 127 && e > 127 {
						inter++
					}
					if g > 127 || e > 127 {
						union++
					}
					d := int(g) - int(e)
					if d < 0 {
						d = -d
					}
					diff += d
					n++
				}
			}
			iou := float64(inter) / float64(union)
			mae := float64(diff) / float64(n) / 255.0
			t.Logf("%dx%d: foreground IoU %.4f, mean abs error %.4f", w, h, iou, mae)
			if iou < 0.85 {
				t.Errorf("foreground IoU %.4f below 0.85", iou)
			}
			if mae > 0.10 {
				t.Errorf("mean absolute error %.4f above 0.10", mae)
			}
		})
	}
}

// TestRemoveKeepsSize checks the cutout geometry and that the matte is not a
// constant (a model returning all-zero alpha would otherwise pass).
func TestRemoveKeepsSize(t *testing.T) {
	skipWithoutModel(t)
	input, _, w, h := splitPanels(t, filepath.Join("testdata", "example1.png"))
	r, err := New(Config{ModelPath: filepath.Join("testdata", "withoutbg-open-weights.onnx")})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	cut, err := r.Remove(input)
	if err != nil {
		t.Fatal(err)
	}
	if cut.Bounds().Dx() != w || cut.Bounds().Dy() != h {
		t.Fatalf("cutout size %v, want %dx%d", cut.Bounds(), w, h)
	}
	var opaque, transparent int
	for y := 0; y < h; y += 3 {
		for x := 0; x < w; x += 3 {
			switch a := cut.NRGBAAt(x, y).A; {
			case a > 250:
				opaque++
			case a < 5:
				transparent++
			}
		}
	}
	t.Logf("sampled pixels: %d opaque, %d transparent", opaque, transparent)
	if opaque == 0 || transparent == 0 {
		t.Errorf("matte looks constant (opaque=%d transparent=%d)", opaque, transparent)
	}
	// RGB must be preserved where the matte is opaque.
	c := cut.NRGBAAt(w/2, h/2)
	o := input.RGBAAt(w/2, h/2)
	if c.A == 255 && (c.R != o.R || c.G != o.G || c.B != o.B) {
		t.Errorf("opaque pixel changed colour: got %v want %v", c, o)
	}
	_ = color.RGBA{}
}
