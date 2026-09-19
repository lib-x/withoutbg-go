// Command withoutbg removes the background from an image with the withoutBG
// open-weights ONNX model.
//
//	withoutbg -model withoutbg-open-weights.onnx -o cutout.png photo.jpg
//	withoutbg -model withoutbg-open-weights.onnx -alpha mask.png photo.jpg
//
// The sidecar (<model>.onnx.json) is read automatically; pass -sidecar to point
// at a different one. The ONNX Runtime shared library is resolved through
// ONNXRUNTIME_LIB_PATH or downloaded by the pure-onnx bootstrap.
package main

import (
	"flag"
	"fmt"
	"image"
	"log"
	"os"
	"time"

	"github.com/lib-x/withoutbg-go"
)

func main() {
	modelPath := flag.String("model", "", "path to withoutbg-open-weights.onnx")
	sidecarPath := flag.String("sidecar", "", "path to the sidecar JSON (default: <model>.json)")
	outPath := flag.String("o", "cutout.png", "output PNG path for the cutout")
	alphaPath := flag.String("alpha", "", "optional path to also write the alpha matte as a grayscale PNG")
	verify := flag.Bool("verify-sha256", false, "verify the model file against the sidecar sha256 before loading")
	flag.Parse()
	if *modelPath == "" || flag.NArg() != 1 {
		flag.Usage()
		os.Exit(2)
	}

	in, err := os.Open(flag.Arg(0))
	if err != nil {
		log.Fatalf("open image: %v", err)
	}
	defer in.Close()

	start := time.Now()
	r, err := withoutbg.New(withoutbg.Config{
		ModelPath:         *modelPath,
		SidecarPath:       *sidecarPath,
		VerifyModelSHA256: *verify,
	})
	if err != nil {
		log.Fatalf("load model: %v", err)
	}
	defer r.Close()
	log.Printf("contract: %s", r.Sidecar())

	// Read the image once, then run the model on it. The streaming helpers
	// accept any io.Reader / io.Writer, so callers can pipe files, HTTP bodies
	// or buffers straight through.
	img, format, err := withoutbg.DecodeImage(in)
	if err != nil {
		log.Fatalf("decode image: %v", err)
	}
	log.Printf("input %s %dx%d", format, img.Bounds().Dx(), img.Bounds().Dy())

	cutout, err := r.Remove(img)
	if err != nil {
		log.Fatalf("remove background: %v", err)
	}
	if err := writePNG(*outPath, cutout); err != nil {
		log.Fatalf("write cutout: %v", err)
	}
	fmt.Printf("%s\t%dms\n", *outPath, time.Since(start).Milliseconds())

	if *alphaPath != "" {
		alpha, err := r.Alpha(img)
		if err != nil {
			log.Fatalf("alpha: %v", err)
		}
		if err := writePNG(*alphaPath, alpha); err != nil {
			log.Fatalf("write alpha: %v", err)
		}
		fmt.Println(*alphaPath)
	}
}

func writePNG(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return withoutbg.WritePNG(f, img)
}
