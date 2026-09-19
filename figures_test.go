package withoutbg

import (
	"image"
	"image/color"
	"image/draw"
	"image/png"
	_ "image/png"
	"os"
	"path/filepath"
	"testing"
)

// TestGenerateDocsFigure writes docs/example-cutout.png: a three-panel strip
// (input | cutout over a checkerboard | alpha matte) produced by this package
// from the published sample. Opt-in because it writes into the working tree:
//
//	WITHOUTBG_UPDATE_FIGURES=1 go test -run TestGenerateDocsFigure .
func TestGenerateDocsFigure(t *testing.T) {
	if os.Getenv("WITHOUTBG_UPDATE_FIGURES") == "" {
		t.Skip("set WITHOUTBG_UPDATE_FIGURES=1 to regenerate docs/example-cutout.png")
	}
	skipWithoutModel(t)
	input, _, w, h := splitPanels(t, filepath.Join("testdata", "example2.png"))

	r, err := New(Config{ModelPath: filepath.Join("testdata", "withoutbg-open-weights.onnx")})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	cutout, err := r.Remove(input)
	if err != nil {
		t.Fatal(err)
	}
	alpha, err := r.Alpha(input)
	if err != nil {
		t.Fatal(err)
	}

	const gap = 12
	canvas := image.NewRGBA(image.Rect(0, 0, w*3+gap*2, h))
	draw.Draw(canvas, image.Rect(0, 0, w, h), input, image.Point{}, draw.Src)
	// Middle panel: the cutout over a checkerboard so transparency is visible.
	mid := image.Rect(w+gap, 0, 2*w+gap, h)
	draw.Draw(canvas, mid, checkerboard(w, h), image.Point{}, draw.Src)
	draw.DrawMask(canvas, mid, cutout, image.Point{}, cutout, image.Point{}, draw.Over)
	// Right panel: the matte as grayscale.
	draw.Draw(canvas, image.Rect(2*(w+gap), 0, 2*(w+gap)+w, h), alpha, image.Point{}, draw.Src)

	if err := os.MkdirAll("docs", 0o755); err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(filepath.Join("docs", "example-cutout.png"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := png.Encode(f, canvas); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote docs/example-cutout.png (%dx%d)", canvas.Bounds().Dx(), canvas.Bounds().Dy())
}

func checkerboard(w, h int) *image.RGBA {
	const cell = 16
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	light := color.RGBA{R: 230, G: 230, B: 230, A: 255}
	dark := color.RGBA{R: 200, G: 200, B: 200, A: 255}
	for y := range h {
		for x := range w {
			c := light
			if (x/cell+y/cell)%2 == 1 {
				c = dark
			}
			img.SetRGBA(x, y, c)
		}
	}
	return img
}
