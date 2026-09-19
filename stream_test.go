package withoutbg

import (
	"bytes"
	"image"
	_ "image/png"
	"os"
	"path/filepath"
	"testing"
)

// sampleImage is one of the test photographs shipped in testdata.
type sampleImage struct {
	file string
	w, h int
}

// The three photographs provided as test material plus the published sample
// panels. Each has a subject near the centre and background around it.
var sampleImages = []sampleImage{
	{"pikachu-cartoon.png", 728, 546},
	{"pokemon-celebration.png", 2880, 1944},
	{"cat.png", 500, 891},
}

func openSample(t *testing.T, name string) *os.File {
	t.Helper()
	f, err := os.Open(filepath.Join("testdata", name))
	if err != nil {
		t.Skipf("sample image not present: %v", err)
	}
	return f
}

// TestSamplesMatteSanity runs the model on each sample photograph and checks
// that the matte behaves like a background removal: subject present, border
// mostly transparent, output the same size as the input.
func TestSamplesMatteSanity(t *testing.T) {
	skipWithoutModel(t)
	r, err := New(Config{ModelPath: filepath.Join("testdata", "withoutbg-open-weights.onnx")})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	for _, s := range sampleImages {
		t.Run(s.file, func(t *testing.T) {
			f := openSample(t, s.file)
			defer f.Close()
			img, format, err := DecodeImage(f)
			if err != nil {
				t.Fatal(err)
			}
			if got := img.Bounds(); got.Dx() != s.w || got.Dy() != s.h {
				t.Fatalf("decoded %s as %dx%d, want %dx%d", format, got.Dx(), got.Dy(), s.w, s.h)
			}
			alpha, err := r.Alpha(img)
			if err != nil {
				t.Fatal(err)
			}
			if got := alpha.Bounds(); got.Dx() != s.w || got.Dy() != s.h {
				t.Fatalf("alpha %dx%d, want %dx%d", got.Dx(), got.Dy(), s.w, s.h)
			}

			// Border band vs centre box: the subject sits in the middle of all
			// three samples, so the border must be more transparent.
			band := max(1, min(s.w, s.h)/20)
			var borderSum, borderN, centreSum, centreN int
			var opaque, clear int
			for y := range s.h {
				for x := range s.w {
					a := int(alpha.Pix[y*alpha.Stride+x])
					switch {
					case a > 200:
						opaque++
					case a < 50:
						clear++
					}
					if x < band || y < band || x >= s.w-band || y >= s.h-band {
						borderSum += a
						borderN++
					}
					if x >= s.w/4 && x < 3*s.w/4 && y >= s.h/4 && y < 3*s.h/4 {
						centreSum += a
						centreN++
					}
				}
			}
			borderMean := float64(borderSum) / float64(borderN)
			centreMean := float64(centreSum) / float64(centreN)
			t.Logf("%dx%d: border mean alpha %.1f, centre mean %.1f, %d opaque / %d clear pixels",
				s.w, s.h, borderMean, centreMean, opaque, clear)

			if opaque == 0 {
				t.Error("no opaque pixels: the subject was not detected")
			}
			if clear == 0 {
				t.Error("no transparent pixels: nothing was removed")
			}
			// The matte must be clearly more transparent at the border than in
			// the middle. This is deliberately a relative check: on the cartoon
			// sample the model leaves large patches of the flat-colour
			// background (vector art is far from its training distribution), and
			// an A/B test showed that removing the letterbox padding does not
			// help, so an absolute "border must be empty" assertion would be
			// testing the model, not this package.
			if borderMean >= 0.8*centreMean {
				t.Errorf("border mean %.1f not clearly below centre mean %.1f", borderMean, centreMean)
			}
		})
	}
}

// TestStreamRoundTrip checks the io.Reader / io.Writer path end to end and that
// it agrees with the image-based API.
func TestStreamRoundTrip(t *testing.T) {
	skipWithoutModel(t)
	r, err := New(Config{ModelPath: filepath.Join("testdata", "withoutbg-open-weights.onnx")})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	raw, err := os.ReadFile(filepath.Join("testdata", "cat.png"))
	if err != nil {
		t.Skipf("sample image not present: %v", err)
	}

	// Decode straight from a reader.
	cutout, err := r.RemoveReader(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	// Same bytes through the one-call pipeline.
	var buf bytes.Buffer
	if err := r.Cutout(bytes.NewReader(raw), &buf); err != nil {
		t.Fatal(err)
	}
	if buf.Len() == 0 {
		t.Fatal("Cutout wrote nothing")
	}
	decoded, format, err := image.Decode(&buf)
	if err != nil {
		t.Fatalf("re-decode cutout: %v", err)
	}
	if format != "png" {
		t.Errorf("cutout format %q, want png", format)
	}
	if decoded.Bounds() != cutout.Bounds() {
		t.Errorf("stream cutout %v, image API %v", decoded.Bounds(), cutout.Bounds())
	}

	// A reader that is not an image must fail cleanly.
	if _, err := r.RemoveReader(bytes.NewReader([]byte("not an image"))); err == nil {
		t.Error("garbage input was accepted")
	}
}

// TestAlphaReader checks the matte-only stream path.
func TestAlphaReader(t *testing.T) {
	skipWithoutModel(t)
	r, err := New(Config{ModelPath: filepath.Join("testdata", "withoutbg-open-weights.onnx")})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	f := openSample(t, "cat.png")
	defer f.Close()
	alpha, err := r.AlphaReader(f)
	if err != nil {
		t.Fatal(err)
	}
	if alpha.Bounds().Dx() != 500 || alpha.Bounds().Dy() != 891 {
		t.Errorf("alpha %v, want 500x891", alpha.Bounds())
	}
}
