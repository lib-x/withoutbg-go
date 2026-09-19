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

// TestGenerateTestImageFigure writes docs/test-images-cutout.png: the three
// test photographs on top, their cutouts (over a checkerboard) below. Opt-in:
//
//	WITHOUTBG_UPDATE_FIGURES=1 go test -run TestGenerateTestImageFigure .
func TestGenerateTestImageFigure(t *testing.T) {
	if os.Getenv("WITHOUTBG_UPDATE_FIGURES") == "" {
		t.Skip("set WITHOUTBG_UPDATE_FIGURES=1 to regenerate docs/test-images-cutout.png")
	}
	skipWithoutModel(t)
	r, err := New(Config{ModelPath: filepath.Join("testdata", "withoutbg-open-weights.onnx")})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	const box, gap = 400, 24
	type cell struct {
		input, cutout *image.NRGBA
		w             int
	}
	cells := make([]cell, 0, len(sampleImages))
	for _, s := range sampleImages {
		f, err := os.Open(filepath.Join("testdata", s.file))
		if err != nil {
			t.Fatal(err)
		}
		img, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			t.Fatal(err)
		}
		cut, err := r.Remove(img)
		if err != nil {
			t.Fatal(err)
		}
		cells = append(cells, cell{input: fitBox(img, box), cutout: fitBox(cut, box), w: fitBox(img, box).Bounds().Dx()})
	}

	width := gap
	for _, c := range cells {
		width += c.w + gap
	}
	canvas := image.NewRGBA(image.Rect(0, 0, width, 2*box+3*gap))
	fillChecker(canvas, color.RGBA{R: 236, G: 236, B: 236, A: 255})
	x := gap
	for _, c := range cells {
		yIn := gap + (box-c.input.Bounds().Dy())/2
		draw.Draw(canvas, image.Rect(x, yIn, x+c.w, yIn+c.input.Bounds().Dy()), c.input, image.Point{}, draw.Src)
		yOut := 2*gap + box + (box-c.cutout.Bounds().Dy())/2
		draw.DrawMask(canvas, image.Rect(x, yOut, x+c.w, yOut+c.cutout.Bounds().Dy()),
			c.cutout, image.Point{}, c.cutout, image.Point{}, draw.Over)
		x += c.w + gap
	}

	f, err := os.Create(filepath.Join("docs", "test-images-cutout.png"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := png.Encode(f, canvas); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote docs/test-images-cutout.png (%dx%d)", canvas.Bounds().Dx(), canvas.Bounds().Dy())
}

// fitBox scales an image so its longest side is box, preserving the aspect
// ratio. Colours are interpolated in premultiplied space so cutouts keep their
// soft edges.
func fitBox(src image.Image, box int) *image.NRGBA {
	b := src.Bounds()
	sw, sh := b.Dx(), b.Dy()
	w, h := box, box
	if sw >= sh {
		h = max(1, int(float64(box)*float64(sh)/float64(sw)+0.5))
	} else {
		w = max(1, int(float64(box)*float64(sw)/float64(sh)+0.5))
	}
	out := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		fy := (float64(y)+0.5)*float64(sh)/float64(h) - 0.5
		y0 := clampIndex(fy, sh)
		y1 := min(y0+1, sh-1)
		wy := weight(fy)
		for x := range w {
			fx := (float64(x)+0.5)*float64(sw)/float64(w) - 0.5
			x0 := clampIndex(fx, sw)
			x1 := min(x0+1, sw-1)
			wx := weight(fx)
			var r, g, bl, a float32
			for _, s := range []struct {
				px, py int
				wt     float32
			}{
				{x0, y0, (1 - wx) * (1 - wy)},
				{x1, y0, wx * (1 - wy)},
				{x0, y1, (1 - wx) * wy},
				{x1, y1, wx * wy},
			} {
				cr, cg, cb, ca := src.At(b.Min.X+s.px, b.Min.Y+s.py).RGBA() // premultiplied
				r += float32(cr) * s.wt
				g += float32(cg) * s.wt
				bl += float32(cb) * s.wt
				a += float32(ca) * s.wt
			}
			out.SetNRGBA(x, y, unpremultiply(r, g, bl, a))
		}
	}
	return out
}

// unpremultiply converts 16-bit premultiplied floats back to an NRGBA pixel.
func unpremultiply(r, g, b, a float32) color.NRGBA {
	if a <= 0 {
		return color.NRGBA{}
	}
	clamp := func(v float32) uint8 {
		v = v * 255 / a
		if v <= 0 {
			return 0
		}
		if v >= 255 {
			return 255
		}
		return uint8(v + 0.5)
	}
	return color.NRGBA{R: clamp(r), G: clamp(g), B: clamp(b), A: uint8(min(a/257+0.5, 255))}
}

func fillChecker(img *image.RGBA, c color.RGBA) {
	for y := img.Bounds().Min.Y; y < img.Bounds().Max.Y; y++ {
		for x := img.Bounds().Min.X; x < img.Bounds().Max.X; x++ {
			img.SetRGBA(x, y, c)
		}
	}
}
