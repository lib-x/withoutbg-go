package withoutbg

import (
	"fmt"
	"image"
	_ "image/jpeg" // registering the decoders here lets callers feed PNG/JPEG streams directly
	"image/png"
	"io"
)

// DecodeImage reads an image from rd. PNG and JPEG are supported out of the
// box; other formats work if the caller has registered them with image.
func DecodeImage(rd io.Reader) (image.Image, string, error) {
	img, format, err := image.Decode(rd)
	if err != nil {
		return nil, "", fmt.Errorf("withoutbg: decode image: %w", err)
	}
	return img, format, nil
}

// RemoveReader decodes an image from rd and returns the cutout.
func (r *Remover) RemoveReader(rd io.Reader) (*image.NRGBA, error) {
	img, _, err := DecodeImage(rd)
	if err != nil {
		return nil, err
	}
	return r.Remove(img)
}

// AlphaReader decodes an image from rd and returns the matte at its original
// size.
func (r *Remover) AlphaReader(rd io.Reader) (*image.Gray, error) {
	img, _, err := DecodeImage(rd)
	if err != nil {
		return nil, err
	}
	return r.Alpha(img)
}

// Cutout is the whole stream pipeline in one call: decode an image from rd,
// remove its background, and write the PNG cutout to w. Use RemoveReader or
// AlphaReader when you want the images instead of encoded bytes.
func (r *Remover) Cutout(rd io.Reader, w io.Writer) error {
	cutout, err := r.RemoveReader(rd)
	if err != nil {
		return err
	}
	return WritePNG(w, cutout)
}

// WritePNG encodes img as PNG into w. The cutouts are RGBA, so the alpha
// channel is preserved.
func WritePNG(w io.Writer, img image.Image) error {
	if err := png.Encode(w, img); err != nil {
		return fmt.Errorf("withoutbg: encode png: %w", err)
	}
	return nil
}
