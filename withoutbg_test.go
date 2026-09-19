package withoutbg

import (
	"image"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadSidecar(t *testing.T) {
	// The published sidecar, when present.
	if _, err := os.Stat(filepath.Join("testdata", "withoutbg-open-weights.onnx.json")); err == nil {
		s, err := LoadSidecar(filepath.Join("testdata", "withoutbg-open-weights.onnx.json"))
		if err != nil {
			t.Fatal(err)
		}
		if s.CanvasSize != 448 {
			t.Errorf("canvas = %d, want 448", s.CanvasSize)
		}
		if s.InputName != "rgb" || s.OutputName != "alpha" {
			t.Errorf("tensor names = %s/%s, want rgb/alpha", s.InputName, s.OutputName)
		}
		if len(s.InputShape) != 4 || s.InputShape[0] != 1 || s.InputShape[1] != 3 {
			t.Errorf("input shape = %v, want [1 3 448 448]", s.InputShape)
		}
		if s.SHA256 == "" {
			t.Error("sidecar carries no sha256")
		}
	}

	// A minimal hand-written sidecar.
	dir := t.TempDir()
	path := filepath.Join(dir, "mini.onnx.json")
	if err := os.WriteFile(path, []byte(`{"canvas_size":448,"input_name":"rgb","output_name":"alpha","input_dtype":"float32"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	s, err := LoadSidecar(path)
	if err != nil {
		t.Fatal(err)
	}
	if s.CanvasSize != 448 || s.InputName != "rgb" {
		t.Errorf("parsed %+v", s)
	}

	// Missing canvas size must fail: it would otherwise be guessed.
	bad := filepath.Join(dir, "bad.onnx.json")
	if err := os.WriteFile(bad, []byte(`{"input_name":"rgb","output_name":"alpha"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadSidecar(bad); err == nil {
		t.Error("sidecar without canvas_size was accepted")
	}

	// Non-float32 input dtype is rejected rather than silently mis-scaled.
	wrong := filepath.Join(dir, "fp16.onnx.json")
	if err := os.WriteFile(wrong, []byte(`{"canvas_size":448,"input_name":"rgb","output_name":"alpha","input_dtype":"float16"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadSidecar(wrong); err == nil {
		t.Error("float16 sidecar was accepted")
	}
}

func TestFitLetterbox(t *testing.T) {
	cases := []struct {
		w, h, canvas int
		wantW, wantH int
	}{
		{400, 491, 448, 365, 448},  // portrait: longest side to 448
		{1000, 500, 448, 448, 224}, // landscape
		{448, 448, 448, 448, 448},  // already square
		{1, 1000, 448, 1, 448},     // degenerate width stays >= 1
	}
	for _, c := range cases {
		got := fit(c.w, c.h, c.canvas)
		if got.newW != c.wantW || got.newH != c.wantH {
			t.Errorf("fit(%d,%d,%d) = %dx%d, want %dx%d", c.w, c.h, c.canvas, got.newW, got.newH, c.wantW, c.wantH)
		}
		if got.newW > c.canvas || got.newH > c.canvas {
			t.Errorf("fit(%d,%d,%d) exceeds the canvas", c.w, c.h, c.canvas)
		}
	}
}

func TestInputTensorLetterboxAndScale(t *testing.T) {
	// A 2x1 image: red on the left, blue on the right. Canvas 4 => the image
	// occupies 4x2 in the top-left, the rest stays black (0).
	img := image.NewRGBA(image.Rect(0, 0, 2, 1))
	img.SetRGBA(0, 0, color.RGBA{R: 255, A: 255})
	img.SetRGBA(1, 0, color.RGBA{B: 255, A: 255})

	r := &Remover{cfg: Config{CanvasSize: 4}}
	data, box, err := r.inputTensor(img)
	if err != nil {
		t.Fatal(err)
	}
	if box.newW != 4 || box.newH != 2 {
		t.Fatalf("letterbox = %dx%d, want 4x2", box.newW, box.newH)
	}
	const plane = 4 * 4
	at := func(c, x, y int) float32 { return data[c*plane+y*4+x] }

	// Top-left pixel: pure red => R=1, G=B=0.
	if math.Abs(float64(at(0, 0, 0))-1) > 1e-6 || at(1, 0, 0) != 0 || at(2, 0, 0) != 0 {
		t.Errorf("pixel (0,0) = %v %v %v, want 1 0 0", at(0, 0, 0), at(1, 0, 0), at(2, 0, 0))
	}
	// Top-right pixel: pure blue => B=1.
	if math.Abs(float64(at(2, 3, 0))-1) > 1e-6 {
		t.Errorf("pixel (3,0) blue = %v, want 1", at(2, 3, 0))
	}
	// Below the image (padding): all channels 0.
	for c := range 3 {
		if at(c, 0, 3) != 0 {
			t.Errorf("padding pixel channel %d = %v, want 0", c, at(c, 0, 3))
		}
	}
}

func TestAlphaFromOutputCropsAndScales(t *testing.T) {
	// Canvas 4, image 2x1 => the matte lives in the top-left 4x2 region; the
	// bottom half of the canvas must be ignored.
	const canvas = 4
	out := make([]float32, canvas*canvas)
	for y := range 2 {
		for x := range canvas {
			out[y*canvas+x] = 1 // foreground in the letterboxed region
		}
	}
	// Bottom rows are 0 (padding); a naive full-canvas resize would halve the
	// alpha, so this checks the crop.
	box := letterbox{srcW: 2, srcH: 1, newW: 4, newH: 2, scale: 2}
	alpha := alphaFromOutput(out, canvas, box)
	if alpha.Bounds().Dx() != 2 || alpha.Bounds().Dy() != 1 {
		t.Fatalf("alpha bounds %v, want 2x1", alpha.Bounds())
	}
	for x := range 2 {
		if v := alpha.Pix[x]; v != 255 {
			t.Errorf("alpha[%d] = %d, want 255", x, v)
		}
	}
}

func TestQuantizeAlpha(t *testing.T) {
	cases := []struct {
		in   float32
		want uint8
	}{
		{-1, 0}, {0, 0}, {0.5, 128}, {1, 255}, {2, 255},
	}
	for _, c := range cases {
		if got := quantizeAlpha(c.in); got != c.want {
			t.Errorf("quantizeAlpha(%v) = %d, want %d", c.in, got, c.want)
		}
	}
}

func TestCompositeAttachesAlpha(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 2, 1))
	img.SetRGBA(0, 0, color.RGBA{R: 10, G: 20, B: 30, A: 255})
	img.SetRGBA(1, 0, color.RGBA{R: 40, G: 50, B: 60, A: 255})
	alpha := image.NewGray(image.Rect(0, 0, 2, 1))
	alpha.Pix[0], alpha.Pix[1] = 255, 0

	out := composite(img, alpha)
	if c := out.RGBAAt(0, 0); c.A != 255 || c.R != 10 || c.G != 20 || c.B != 30 {
		t.Errorf("opaque pixel = %v, want RGBA(10,20,30,255)", c)
	}
	if c := out.RGBAAt(1, 0); c.A != 0 || c.R != 40 {
		t.Errorf("transparent pixel = %v, want RGB kept with A=0", c)
	}
}
