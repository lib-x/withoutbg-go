package withoutbg

import (
	"image"
	"image/color"
)

// alphaFromOutput turns the model's canvas-sized matte into an image at the
// original size: crop to the letterboxed region (top-left, before padding),
// resize back, and quantise to 8 bits. The reference postprocessing does
// exactly these three steps.
func alphaFromOutput(data []float32, canvas int, box letterbox) *image.Gray {
	crop := make([]float32, box.newW*box.newH)
	for y := range box.newH {
		src := y * canvas
		dst := y * box.newW
		copy(crop[dst:dst+box.newW], data[src:src+box.newW])
	}
	out := image.NewGray(image.Rect(0, 0, box.srcW, box.srcH))
	for y := range box.srcH {
		fy := (float64(y)+0.5)*float64(box.newH)/float64(box.srcH) - 0.5
		y0 := clampIndex(fy, box.newH)
		y1 := min(y0+1, box.newH-1)
		wy := weight(fy)
		for x := range box.srcW {
			fx := (float64(x)+0.5)*float64(box.newW)/float64(box.srcW) - 0.5
			x0 := clampIndex(fx, box.newW)
			x1 := min(x0+1, box.newW-1)
			wx := weight(fx)
			v := crop[y0*box.newW+x0]*(1-wx)*(1-wy) +
				crop[y0*box.newW+x1]*wx*(1-wy) +
				crop[y1*box.newW+x0]*(1-wx)*wy +
				crop[y1*box.newW+x1]*wx*wy
			out.Pix[y*out.Stride+x] = quantizeAlpha(v)
		}
	}
	return out
}

// composite copies the source image and attaches the matte as its alpha
// channel, which is what "cutout PNG" means.
func composite(img image.Image, alpha *image.Gray) *image.RGBA {
	b := img.Bounds()
	out := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := range b.Dy() {
		for x := range b.Dx() {
			r, g, bl, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			a := alpha.Pix[y*alpha.Stride+x]
			out.SetRGBA(x, y, color.RGBA{
				R: uint8(r >> 8),
				G: uint8(g >> 8),
				B: uint8(bl >> 8),
				A: a,
			})
		}
	}
	return out
}

func clampIndex(f float64, n int) int {
	i := int(f)
	if i < 0 {
		return 0
	}
	if i > n-1 {
		return n - 1
	}
	return i
}

func weight(f float64) float32 {
	w := f - float64(int(f))
	if w < 0 {
		return 0
	}
	return float32(w)
}

func quantizeAlpha(v float32) uint8 {
	switch {
	case v <= 0:
		return 0
	case v >= 1:
		return 255
	default:
		return uint8(v*255 + 0.5)
	}
}
