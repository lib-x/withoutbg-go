package withoutbg

import (
	"errors"
	"fmt"
	"image"
)

// Config controls model loading and preprocessing. Zero values fall back to the
// values from the sidecar (and then to the package defaults).
type Config struct {
	// ModelPath is the path to withoutbg-open-weights.onnx (required).
	ModelPath string

	// SidecarPath overrides the sidecar location. Empty means
	// ModelPath + ".json", which is how the published export ships.
	SidecarPath string

	// CanvasSize is the square letterbox canvas fed to the model. Empty/zero
	// means "take it from the sidecar".
	CanvasSize int

	// InputName / OutputName override the ONNX tensor names. Empty means
	// "take them from the sidecar", falling back to the graph itself.
	InputName  string
	OutputName string

	// VerifyModelSHA256 checks the model file against the sidecar's sha256
	// before loading. It reads the whole file (455 MB for the published
	// export), so it is off by default.
	VerifyModelSHA256 bool

	// LibraryPath optionally points to a prebuilt libonnxruntime.so. It is
	// used only when the runtime is not yet initialized; otherwise the
	// ONNXRUNTIME_LIB_PATH environment variable or the pure-onnx bootstrap
	// download applies.
	LibraryPath string

	sidecar *Sidecar
}

// applyDefaults fills zero fields from the sidecar and package defaults. It is
// separate from validation so callers can inspect the effective config.
func (c *Config) applyDefaults() {
	if c.SidecarPath == "" && c.ModelPath != "" {
		c.SidecarPath = SidecarPath(c.ModelPath)
	}
	if c.CanvasSize == 0 {
		if c.sidecar != nil && c.sidecar.CanvasSize > 0 {
			c.CanvasSize = c.sidecar.CanvasSize
		} else {
			c.CanvasSize = DefaultCanvasSize
		}
	}
	if c.InputName == "" {
		if c.sidecar != nil && c.sidecar.InputName != "" {
			c.InputName = c.sidecar.InputName
		} else {
			c.InputName = DefaultInputName
		}
	}
	if c.OutputName == "" {
		if c.sidecar != nil && c.sidecar.OutputName != "" {
			c.OutputName = c.sidecar.OutputName
		} else {
			c.OutputName = DefaultOutputName
		}
	}
}

// validate checks the fields a caller must supply.
func (c *Config) validate() error {
	if c.ModelPath == "" {
		return errors.New("withoutbg: Config.ModelPath is required")
	}
	if c.CanvasSize <= 0 {
		return fmt.Errorf("withoutbg: invalid CanvasSize %d", c.CanvasSize)
	}
	if c.InputName == "" || c.OutputName == "" {
		return errors.New("withoutbg: InputName and OutputName must not be empty")
	}
	return nil
}

// letterbox describes how an image was fitted onto the square canvas: the
// resized size in the top-left corner and the scale factor applied.
type letterbox struct {
	srcW, srcH int
	newW, newH int
	scale      float64
}

// fit computes the letterbox geometry used by the reference preprocessing:
// the longest side is scaled to canvas, the aspect ratio is preserved, and the
// result is pasted at the top-left of a black canvas.
func fit(srcW, srcH, canvas int) letterbox {
	scale := float64(canvas) / float64(max(srcW, srcH))
	newW := max(1, int(round(float64(srcW)*scale)))
	newH := max(1, int(round(float64(srcH)*scale)))
	return letterbox{srcW: srcW, srcH: srcH, newW: newW, newH: newH, scale: scale}
}

func round(f float64) float64 { return float64(int(f + 0.5)) }

// bounds is a small helper used by tests and the CLI.
func imageSize(img image.Image) (int, int) {
	b := img.Bounds()
	return b.Dx(), b.Dy()
}
