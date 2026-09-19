package withoutbg

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"image"
	"io"
	"os"
	"sync"

	ort "github.com/amikos-tech/pure-onnx/ort"

	"github.com/lib-x/withoutbg-go/internal/onnxmeta"
)

// Remover runs the withoutBG matting model.
type Remover struct {
	cfg     Config
	sidecar *Sidecar
	sess    *ort.AdvancedSession
	closed  bool

	mu sync.Mutex // guards sess use and closed
}

var (
	envInitSet bool // true once the library has initialized the runtime
	envMu      sync.Mutex
	refCount   int // open Remover count, gates DestroyEnvironment
)

// New loads the sidecar, validates the ONNX graph against it and prepares an
// inference session.
//
// The sidecar is required: it carries the canvas size and tensor names, and the
// model repository documents it as the authoritative contract. Loading a model
// without it would mean guessing the preprocessing.
func New(cfg Config) (*Remover, error) {
	if cfg.ModelPath == "" {
		return nil, errors.New("withoutbg: Config.ModelPath is required")
	}
	if cfg.SidecarPath == "" {
		cfg.SidecarPath = SidecarPath(cfg.ModelPath)
	}
	sidecar, err := LoadSidecar(cfg.SidecarPath)
	if err != nil {
		return nil, err
	}
	cfg.sidecar = sidecar
	cfg.applyDefaults()
	if err := cfg.validate(); err != nil {
		return nil, err
	}

	meta, err := onnxmeta.ParseFile(cfg.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("withoutbg: inspect model: %w", err)
	}
	if err := validateContract(meta, sidecar, &cfg); err != nil {
		return nil, err
	}

	if cfg.VerifyModelSHA256 {
		if err := verifySHA256(cfg.ModelPath, sidecar.SHA256); err != nil {
			return nil, err
		}
	}

	if err := initEnvironment(cfg.LibraryPath); err != nil {
		return nil, err
	}
	envMu.Lock()
	refCount++
	envMu.Unlock()

	canvas := int64(cfg.CanvasSize)
	inShape := ort.Shape{1, 3, canvas, canvas}
	outShape := ort.Shape{1, 1, canvas, canvas}
	placeholderIn, err := ort.NewTensor(inShape, make([]float32, 3*int(canvas)*int(canvas)))
	if err != nil {
		releaseEnvironment()
		return nil, fmt.Errorf("withoutbg: create input placeholder: %w", err)
	}
	placeholderOut, err := ort.NewEmptyTensor[float32](outShape)
	if err != nil {
		placeholderIn.Destroy()
		releaseEnvironment()
		return nil, fmt.Errorf("withoutbg: create output placeholder: %w", err)
	}

	sess, err := ort.NewAdvancedSession(
		cfg.ModelPath,
		[]string{cfg.InputName},
		[]string{cfg.OutputName},
		[]ort.Value{placeholderIn},
		[]ort.Value{placeholderOut},
		nil,
	)
	placeholderIn.Destroy()
	placeholderOut.Destroy()
	if err != nil {
		releaseEnvironment()
		return nil, fmt.Errorf("withoutbg: create session: %w", err)
	}

	return &Remover{cfg: cfg, sidecar: sidecar, sess: sess}, nil
}

// validateContract compares the sidecar against the graph. The sidecar is the
// documented contract, so a mismatch means the wrong pair of files is on disk
// and the preprocessing would silently be wrong.
func validateContract(meta *onnxmeta.Model, sidecar *Sidecar, cfg *Config) error {
	if len(meta.Inputs) != 1 || len(meta.Outputs) != 1 {
		return fmt.Errorf("withoutbg: model has %d inputs and %d outputs, want 1/1",
			len(meta.Inputs), len(meta.Outputs))
	}
	in, out := meta.Inputs[0], meta.Outputs[0]
	if in.Name != cfg.InputName {
		return fmt.Errorf("withoutbg: model input %q does not match sidecar/config %q", in.Name, cfg.InputName)
	}
	if out.Name != cfg.OutputName {
		return fmt.Errorf("withoutbg: model output %q does not match sidecar/config %q", out.Name, cfg.OutputName)
	}
	if len(in.Shape) != 4 || len(out.Shape) != 4 {
		return fmt.Errorf("withoutbg: expected 4-D input/output, got %d/%d", len(in.Shape), len(out.Shape))
	}
	if c := in.Shape[1]; c.Value != 3 {
		return fmt.Errorf("withoutbg: model input channels %d, want 3", c.Value)
	}
	// The published export has a fixed square canvas; check it against the
	// sidecar so a different export cannot slip through unnoticed.
	for _, d := range []struct {
		what string
		dim  onnxmeta.Dim
	}{{"input height", in.Shape[2]}, {"input width", in.Shape[3]}} {
		if d.dim.IsDynamic() {
			continue // dynamic canvas: config/sidecar drives it
		}
		if int(d.dim.Value) != cfg.CanvasSize {
			return fmt.Errorf("withoutbg: model %s %d does not match canvas size %d",
				d.what, d.dim.Value, cfg.CanvasSize)
		}
	}
	if !out.Shape[1].IsDynamic() && out.Shape[1].Value != 1 {
		return fmt.Errorf("withoutbg: model output channels %d, want 1 (alpha)", out.Shape[1].Value)
	}
	if sidecar.MattingInputSize > 0 && sidecar.MattingInputSize != sidecar.CanvasSize {
		return fmt.Errorf("withoutbg: sidecar matting_input_size %d != canvas_size %d",
			sidecar.MattingInputSize, sidecar.CanvasSize)
	}
	return nil
}

// Alpha runs the model and returns the matte at the original image size. White
// (255) means "keep", black (0) means "background".
func (r *Remover) Alpha(img image.Image) (*image.Gray, error) {
	out, box, err := r.run(img)
	if err != nil {
		return nil, err
	}
	return alphaFromOutput(out, r.cfg.CanvasSize, box), nil
}

// Remove returns the input image with the matte attached as its alpha channel,
// ready to be saved as a PNG cutout. The result is an *image.NRGBA, so its RGB
// values are the original colours and A is the matte.
func (r *Remover) Remove(img image.Image) (*image.NRGBA, error) {
	alpha, err := r.Alpha(img)
	if err != nil {
		return nil, err
	}
	return composite(img, alpha), nil
}

// run performs one inference and returns the raw canvas-sized matte.
func (r *Remover) run(img image.Image) ([]float32, letterbox, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, letterbox{}, errors.New("withoutbg: Remover is closed")
	}
	in, box, err := r.inputTensor(img)
	if err != nil {
		return nil, letterbox{}, err
	}
	canvas := int64(r.cfg.CanvasSize)
	inTensor, err := ort.NewTensor(ort.Shape{1, 3, canvas, canvas}, in)
	if err != nil {
		return nil, letterbox{}, fmt.Errorf("withoutbg: input tensor: %w", err)
	}
	defer inTensor.Destroy()
	outTensor, err := ort.NewEmptyTensor[float32](ort.Shape{1, 1, canvas, canvas})
	if err != nil {
		return nil, letterbox{}, fmt.Errorf("withoutbg: output tensor: %w", err)
	}
	defer outTensor.Destroy()

	if err := r.sess.RunWithValues([]ort.Value{inTensor}, []ort.Value{outTensor}); err != nil {
		return nil, letterbox{}, fmt.Errorf("withoutbg: inference: %w", err)
	}
	return outTensor.GetData(), box, nil
}

// Close releases the session. The ONNX Runtime environment is destroyed when
// the last Remover is closed.
func (r *Remover) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil
	}
	r.closed = true
	var errs []error
	if r.sess != nil {
		if err := r.sess.Destroy(); err != nil {
			errs = append(errs, err)
		}
		r.sess = nil
	}
	releaseEnvironment()
	return errors.Join(errs...)
}

// Sidecar returns the loaded contract, useful for logging.
func (r *Remover) Sidecar() *Sidecar { return r.sidecar }

func initEnvironment(libraryPath string) error {
	envMu.Lock()
	defer envMu.Unlock()
	if envInitSet && libraryPath != "" {
		return errors.New("withoutbg: LibraryPath cannot change after the runtime is initialized")
	}
	envInitSet = true
	if libraryPath != "" {
		if err := ort.SetSharedLibraryPath(libraryPath); err != nil {
			return fmt.Errorf("withoutbg: set library path: %w", err)
		}
		_ = ort.SetLogLevel(ort.LoggingLevelFatal)
		return ort.InitializeEnvironment()
	}
	return ort.InitializeEnvironmentWithBootstrap()
}

func releaseEnvironment() {
	envMu.Lock()
	defer envMu.Unlock()
	refCount--
	if refCount <= 0 {
		_ = ort.DestroyEnvironment()
		envInitSet = false
	}
}

// verifySHA256 compares the model file with the hash recorded in the sidecar.
func verifySHA256(path, want string) error {
	if want == "" {
		return errors.New("withoutbg: sidecar has no sha256 to verify against")
	}
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("withoutbg: open model for hashing: %w", err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return fmt.Errorf("withoutbg: hash model: %w", err)
	}
	got := hex.EncodeToString(h.Sum(nil))
	if got != want {
		return fmt.Errorf("withoutbg: model sha256 %s does not match sidecar %s", got, want)
	}
	return nil
}
