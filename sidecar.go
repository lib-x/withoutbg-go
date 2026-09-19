package withoutbg

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// Sidecar is the metadata file that ships next to the ONNX export
// (<model>.onnx.json). It is the authoritative source for the canvas size,
// tensor names and shapes, precision and the model hash; the README of the
// model repository says to read it first, so this package does the same and
// validates the ONNX graph against it.
type Sidecar struct {
	CanvasSize       int     `json:"canvas_size"`
	MattingInputSize int     `json:"matting_input_size"`
	OpsetVersion     int     `json:"opset_version"`
	Precision        string  `json:"precision"`
	Variant          string  `json:"variant"`
	DepthVariant     string  `json:"depth_variant"`
	ConvNeXtSize     string  `json:"convnext_size"`
	ModelVersion     string  `json:"model_version"`
	InputName        string  `json:"input_name"`
	OutputName       string  `json:"output_name"`
	InputDtype       string  `json:"input_dtype"`
	InputShape       []int   `json:"input_shape"`
	OutputShape      []int   `json:"output_shape"`
	SizeMB           float64 `json:"size_mb"`
	SHA256           string  `json:"sha256"`
	// MAE and MaxAbs describe how closely the ONNX export reproduces the
	// reference implementation on the exporter's validation set.
	MAE    float64 `json:"mae"`
	MaxAbs float64 `json:"max_abs"`
}

// SidecarPath returns the conventional sidecar path for a model file
// ("model.onnx" -> "model.onnx.json"). The published export uses exactly this
// naming, so the sidecar is found automatically.
func SidecarPath(modelPath string) string { return modelPath + ".json" }

// LoadSidecar reads and validates a sidecar file. A missing sidecar is an
// error: without it the canvas size and tensor names would be guesses.
func LoadSidecar(path string) (*Sidecar, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("withoutbg: read sidecar: %w", err)
	}
	var s Sidecar
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, fmt.Errorf("withoutbg: parse sidecar %s: %w", path, err)
	}
	if s.CanvasSize <= 0 {
		return nil, fmt.Errorf("withoutbg: sidecar %s has no canvas_size", path)
	}
	if s.InputName == "" || s.OutputName == "" {
		return nil, fmt.Errorf("withoutbg: sidecar %s is missing input_name/output_name", path)
	}
	if s.InputDtype != "" && s.InputDtype != "float32" {
		return nil, fmt.Errorf("withoutbg: sidecar input_dtype %q is not supported (want float32)", s.InputDtype)
	}
	return &s, nil
}

// String summarises the contract for logs and error messages.
func (s *Sidecar) String() string {
	parts := []string{
		fmt.Sprintf("canvas=%d", s.CanvasSize),
		fmt.Sprintf("in=%s%v", s.InputName, s.InputShape),
		fmt.Sprintf("out=%s%v", s.OutputName, s.OutputShape),
	}
	if s.ModelVersion != "" {
		parts = append([]string{"version=" + s.ModelVersion}, parts...)
	}
	if s.Precision != "" {
		parts = append(parts, "precision="+s.Precision)
	}
	return strings.Join(parts, " ")
}
