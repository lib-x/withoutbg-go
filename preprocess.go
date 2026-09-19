package withoutbg

import (
	"fmt"
	"image"
	"image/color"
)

// inputTensor builds the model input: RGB, letterboxed onto a black square
// canvas, scaled to [0,1] and laid out as NCHW float32. This mirrors the
// preprocessing the model repository documents step by step.
func (r *Remover) inputTensor(img image.Image) ([]float32, letterbox, error) {
	srcW, srcH := imageSize(img)
	if srcW <= 0 || srcH <= 0 {
		return nil, letterbox{}, fmt.Errorf("withoutbg: empty image %dx%d", srcW, srcH)
	}
	box := fit(srcW, srcH, r.cfg.CanvasSize)

	rgb := resizeRGBFloat(img, box.newW, box.newH)
	canvas := r.cfg.CanvasSize
	out := make([]float32, 3*canvas*canvas)
	plane := canvas * canvas
	for y := range box.newH {
		rowIn := y * box.newW * 3
		rowOut := y * canvas
		for x := range box.newW {
			i := rowIn + x*3
			out[rowOut+x] = rgb[i]           // R
			out[plane+rowOut+x] = rgb[i+1]   // G
			out[2*plane+rowOut+x] = rgb[i+2] // B
		}
	}
	return out, box, nil
}

// resizeRGBFloat resizes an image to w x h with bilinear interpolation and
// returns interleaved RGB float32 values in [0,1]. Alpha is ignored: the model
// input is a plain RGB photo.
func resizeRGBFloat(img image.Image, w, h int) []float32 {
	b := img.Bounds()
	srcW, srcH := b.Dx(), b.Dy()
	out := make([]float32, w*h*3)
	for y := range h {
		// Pixel-centre mapping, clamped to the source bounds.
		fy := (float64(y)+0.5)*float64(srcH)/float64(h) - 0.5
		y0 := int(fy)
		if fy < 0 {
			y0 = 0
		}
		if y0 > srcH-1 {
			y0 = srcH - 1
		}
		y1 := min(y0+1, srcH-1)
		wy := float32(fy - float64(y0))
		if wy < 0 {
			wy = 0
		}
		for x := range w {
			fx := (float64(x)+0.5)*float64(srcW)/float64(w) - 0.5
			x0 := int(fx)
			if fx < 0 {
				x0 = 0
			}
			if x0 > srcW-1 {
				x0 = srcW - 1
			}
			x1 := min(x0+1, srcW-1)
			wx := float32(fx - float64(x0))
			if wx < 0 {
				wx = 0
			}
			c00 := rgb8(img.At(b.Min.X+x0, b.Min.Y+y0))
			c10 := rgb8(img.At(b.Min.X+x1, b.Min.Y+y0))
			c01 := rgb8(img.At(b.Min.X+x0, b.Min.Y+y1))
			c11 := rgb8(img.At(b.Min.X+x1, b.Min.Y+y1))
			i := (y*w + x) * 3
			for c := range 3 {
				v := c00[c]*(1-wx)*(1-wy) + c10[c]*wx*(1-wy) + c01[c]*(1-wx)*wy + c11[c]*wx*wy
				out[i+c] = v
			}
		}
	}
	return out
}

// rgb8 converts a colour to three [0,1] floats in RGB order.
func rgb8(c color.Color) [3]float32 {
	r, g, b, _ := c.RGBA() // 16-bit premultiplied by alpha
	const inv = 1.0 / 65535.0
	return [3]float32{float32(r) * inv, float32(g) * inv, float32(b) * inv}
}
