// Package withoutbg removes image backgrounds with the withoutBG open-weights
// ONNX model (DepthAnythingV2 + ConvNeXt matting), driven through pure-onnx
// (purego, no CGO).
//
// The model expects a letterboxed RGB tensor at a fixed canvas size and returns
// an alpha matte of the same canvas. The authoritative contract ships next to
// the model file as a sidecar JSON (canvas size, tensor names, shapes, SHA256);
// this package reads it, validates the ONNX graph against it, and uses it for
// the preprocessing defaults.
//
// Typical use:
//
//	r, err := withoutbg.New(withoutbg.Config{ModelPath: "withoutbg-open-weights.onnx"})
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer r.Close()
//
//	cutout, err := r.Remove(img) // *image.RGBA, original size, alpha = matte
package withoutbg

const (
	// DefaultCanvasSize is used when neither the sidecar nor Config provide one.
	DefaultCanvasSize = 448
	// DefaultInputName and DefaultOutputName match the published export.
	DefaultInputName  = "rgb"
	DefaultOutputName = "alpha"
)
