package main

import (
	"fmt"
	"os"
	"time"
	"flag"

	// Package imports
	whisper "github.com/boocmp/whisper.cpp/bindings/go/pkg/whisper"
	wav "github.com/go-audio/wav"
)

func Process(model whisper.Model, path string, flags *Flags) error {
	var data []float32

	// Create processing context
	context, err := model.NewContext()
	if err != nil {
		return err
	}

	// Set the parameters
	if err := flags.SetParams(context); err != nil {
		return err
	}

	fmt.Printf("\n%s\n", context.SystemInfo())

	// Open the file
	fmt.Fprintf(flags.Output(), "Loading %q\n", path)
	fh, err := os.Open(path)
	if err != nil {
		return err
	}
	defer fh.Close()

	// Decode the WAV file - load the full buffer
	dec := wav.NewDecoder(fh)
	if buf, err := dec.FullPCMBuffer(); err != nil {
		return err
	} else if dec.SampleRate != whisper.SampleRate {
		return fmt.Errorf("unsupported sample rate: %d", dec.SampleRate)
	} else if dec.NumChans != 1 {
		return fmt.Errorf("unsupported number of channels: %d", dec.NumChans)
	} else {
		data = buf.AsFloat32Buffer().Data
	}

	// Segment callback when -tokens is specified

	// Process the data

	n := flags.Lookup("states").Value.(flag.Getter).Get().(int)
	quit:= make(chan string, n)

	context.ResetTimings()
	for i := 0; i < n; i++ {
		//go func() error {
			state := context.NewState()
			quit <- fmt.Sprintf("%p %p", state, data)

			/*if segments, err := context.Process(state, data); err != nil {
				quit <- "error"
				return err
			} else {
				result := fmt.Sprintf("%p", state)
				for _, segment := range segments {
					result += " " + segment.Text
				}
				quit <- result
			}*/
		//	return nil
	        //}()
	}

	for i := 0; i < n; i++ {
		fmt.Println(<-quit)
	}

	context.PrintTimings()

	time.Sleep(10 * time.Second)

	return nil
}
