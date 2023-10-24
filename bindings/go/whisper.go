package whisper

import (
	"errors"
	"unsafe"
)

///////////////////////////////////////////////////////////////////////////////
// CGO

/*
#cgo LDFLAGS: -lwhisper -lm -lstdc++ -L/usr/local/cuda/lib64 -lcudart -lcublas
#cgo darwin LDFLAGS: -framework Accelerate
#include <whisper.h>
#include <stdlib.h>
*/
import "C"

///////////////////////////////////////////////////////////////////////////////
// TYPES

type (
	Context          C.struct_whisper_context
	Token            C.whisper_token
	TokenData        C.struct_whisper_token_data
	SamplingStrategy C.enum_whisper_sampling_strategy
	Params           C.struct_whisper_full_params
	State            C.struct_whisper_state
)

///////////////////////////////////////////////////////////////////////////////
// GLOBALS

const (
	SAMPLING_GREEDY      SamplingStrategy = C.WHISPER_SAMPLING_GREEDY
	SAMPLING_BEAM_SEARCH SamplingStrategy = C.WHISPER_SAMPLING_BEAM_SEARCH
)

const (
	SampleRate = C.WHISPER_SAMPLE_RATE                 // Expected sample rate, samples per second
	SampleBits = uint16(unsafe.Sizeof(C.float(0))) * 8 // Sample size in bits
	NumFFT     = C.WHISPER_N_FFT
	NumMEL     = C.WHISPER_N_MEL
	HopLength  = C.WHISPER_HOP_LENGTH
	ChunkSize  = C.WHISPER_CHUNK_SIZE
)

var (
	ErrTokenizerFailed  = errors.New("whisper_tokenize failed")
	ErrAutoDetectFailed = errors.New("whisper_lang_auto_detect failed")
	ErrConversionFailed = errors.New("whisper_convert failed")
	ErrInvalidLanguage  = errors.New("invalid language")
)

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

// Allocates all memory needed for the model and loads the model from the given file.
// Returns NULL on failure.
func Whisper_init(path string) *Context {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	if ctx := C.whisper_init_from_file_no_state(cPath); ctx != nil {
		return (*Context)(ctx)
	} else {
		return nil
	}
}

// Frees all memory allocated by the model.
func (ctx *Context) Whisper_free() {
	C.whisper_free((*C.struct_whisper_context)(ctx))
}

func (ctx *Context) Whisper_init_state() *State {
	if state := C.whisper_init_state((*C.struct_whisper_context)(ctx)); state != nil {
		return (*State)(state)
	} else {
		return nil
	}
}

func (state *State) Close() error {
	if state != nil {
		C.whisper_free_state((*C.struct_whisper_state)(state))
	}
	return nil
}

// Convert RAW PCM audio to log mel spectrogram.
// The resulting spectrogram is stored inside the provided whisper context.
func (ctx *Context) Whisper_pcm_to_mel(data []float32, threads int) error {
	if C.whisper_pcm_to_mel((*C.struct_whisper_context)(ctx), (*C.float)(&data[0]), C.int(len(data)), C.int(threads)) == 0 {
		return nil
	} else {
		return ErrConversionFailed
	}
}

// This can be used to set a custom log mel spectrogram inside the provided whisper context.
// Use this instead of whisper_pcm_to_mel() if you want to provide your own log mel spectrogram.
// n_mel must be 80
func (ctx *Context) Whisper_set_mel(data []float32, n_mel int) error {
	if C.whisper_set_mel((*C.struct_whisper_context)(ctx), (*C.float)(&data[0]), C.int(len(data)), C.int(n_mel)) == 0 {
		return nil
	} else {
		return ErrConversionFailed
	}
}

// Run the Whisper encoder on the log mel spectrogram stored inside the provided whisper context.
// Make sure to call whisper_pcm_to_mel() or whisper_set_mel() first.
// offset can be used to specify the offset of the first frame in the spectrogram.
func (ctx *Context) Whisper_encode(offset, threads int) error {
	if C.whisper_encode((*C.struct_whisper_context)(ctx), C.int(offset), C.int(threads)) == 0 {
		return nil
	} else {
		return ErrConversionFailed
	}
}

// Run the Whisper decoder to obtain the logits and probabilities for the next token.
// Make sure to call whisper_encode() first.
// tokens + n_tokens is the provided context for the decoder.
// n_past is the number of tokens to use from previous decoder calls.
func (ctx *Context) Whisper_decode(tokens []Token, past, threads int) error {
	if C.whisper_decode((*C.struct_whisper_context)(ctx), (*C.whisper_token)(&tokens[0]), C.int(len(tokens)), C.int(past), C.int(threads)) == 0 {
		return nil
	} else {
		return ErrConversionFailed
	}
}

// Convert the provided text into tokens. The tokens pointer must be large enough to hold the resulting tokens.
// Returns the number of tokens on success
func (ctx *Context) Whisper_tokenize(text string, tokens []Token) (int, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))
	if n := C.whisper_tokenize((*C.struct_whisper_context)(ctx), cText, (*C.whisper_token)(&tokens[0]), C.int(len(tokens))); n >= 0 {
		return int(n), nil
	} else {
		return 0, ErrTokenizerFailed
	}
}

// Return the id of the specified language, returns -1 if not found
// Examples:
//
//	"de" -> 2
//	"german" -> 2
func (ctx *Context) Whisper_lang_id(lang string) int {
	return int(C.whisper_lang_id(C.CString(lang)))
}

// Largest language id (i.e. number of available languages - 1)
func Whisper_lang_max_id() int {
	return int(C.whisper_lang_max_id())
}

// Return the short string of the specified language id (e.g. 2 -> "de"),
// returns empty string if not found
func Whisper_lang_str(id int) string {
	return C.GoString(C.whisper_lang_str(C.int(id)))
}

func (ctx *Context) Whisper_n_len() int {
	return int(C.whisper_n_len((*C.struct_whisper_context)(ctx)))
}

func (ctx *Context) Whisper_n_vocab() int {
	return int(C.whisper_n_vocab((*C.struct_whisper_context)(ctx)))
}

func (ctx *Context) Whisper_n_text_ctx() int {
	return int(C.whisper_n_text_ctx((*C.struct_whisper_context)(ctx)))
}

func (ctx *Context) Whisper_n_audio_ctx() int {
	return int(C.whisper_n_audio_ctx((*C.struct_whisper_context)(ctx)))
}

func (ctx *Context) Whisper_is_multilingual() int {
	return int(C.whisper_is_multilingual((*C.struct_whisper_context)(ctx)))
}

// Token Id -> String. Uses the vocabulary in the provided context
func (ctx *Context) Whisper_token_to_str(token Token) string {
	return C.GoString(C.whisper_token_to_str((*C.struct_whisper_context)(ctx), C.whisper_token(token)))
}

// Special tokens
func (ctx *Context) Whisper_token_eot() Token {
	return Token(C.whisper_token_eot((*C.struct_whisper_context)(ctx)))
}

// Special tokens
func (ctx *Context) Whisper_token_sot() Token {
	return Token(C.whisper_token_sot((*C.struct_whisper_context)(ctx)))
}

// Special tokens
func (ctx *Context) Whisper_token_prev() Token {
	return Token(C.whisper_token_prev((*C.struct_whisper_context)(ctx)))
}

// Special tokens
func (ctx *Context) Whisper_token_solm() Token {
	return Token(C.whisper_token_solm((*C.struct_whisper_context)(ctx)))
}

// Special tokens
func (ctx *Context) Whisper_token_not() Token {
	return Token(C.whisper_token_not((*C.struct_whisper_context)(ctx)))
}

// Special tokens
func (ctx *Context) Whisper_token_beg() Token {
	return Token(C.whisper_token_beg((*C.struct_whisper_context)(ctx)))
}

// Special tokens
func (ctx *Context) Whisper_token_lang(lang_id int) Token {
	return Token(C.whisper_token_lang((*C.struct_whisper_context)(ctx), C.int(lang_id)))
}

// Task tokens
func (ctx *Context) Whisper_token_translate() Token {
	return Token(C.whisper_token_translate((*C.struct_whisper_context)(ctx)))
}

// Task tokens
func (ctx *Context) Whisper_token_transcribe() Token {
	return Token(C.whisper_token_transcribe((*C.struct_whisper_context)(ctx)))
}

// Performance information
func (ctx *Context) Whisper_print_timings() {
	C.whisper_print_timings((*C.struct_whisper_context)(ctx))
}

// Performance information
func (ctx *Context) Whisper_reset_timings() {
	C.whisper_reset_timings((*C.struct_whisper_context)(ctx))
}

// Print system information
func Whisper_print_system_info() string {
	return C.GoString(C.whisper_print_system_info())
}

// Return default parameters for a strategy
func (ctx *Context) Whisper_full_default_params(strategy SamplingStrategy) Params {
	// Get default parameters
	return Params(C.whisper_full_default_params(C.enum_whisper_sampling_strategy(strategy)))
}

// Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text
// Uses the specified decoding strategy to obtain the text.
func (ctx *Context) Whisper_full_with_state(
	state *State,
	params Params,
	samples []float32,
) error {
	if C.whisper_full_with_state((*C.struct_whisper_context)(ctx),(*C.struct_whisper_state)(state), (C.struct_whisper_full_params)(params), (*C.float)(&samples[0]), C.int(len(samples))) == 0 {
		return nil
	} else {
		return ErrConversionFailed
	}
}

// Return the id of the autodetected language, returns -1 if not found
// Added to whisper.cpp in
// https://github.com/ggerganov/whisper.cpp/commit/a1c1583cc7cd8b75222857afc936f0638c5683d6
//
// Examples:
//
//	"de" -> 2
//	"german" -> 2
func (state *State) Whisper_full_lang_id() int {
	return int(C.whisper_full_lang_id_from_state((*C.struct_whisper_state)(state)))
}

// Number of generated text segments.
// A segment can be a few words, a sentence, or even a paragraph.
func (state *State) Whisper_full_n_segments() int {
	return int(C.whisper_full_n_segments_from_state((*C.struct_whisper_state)(state)))
}

// Get the start and end time of the specified segment.
func (state *State) Whisper_full_get_segment_t0(segment int) int64 {
	return int64(C.whisper_full_get_segment_t0_from_state((*C.struct_whisper_state)(state), C.int(segment)))
}

// Get the start and end time of the specified segment.
func (state *State) Whisper_full_get_segment_t1(segment int) int64 {
	return int64(C.whisper_full_get_segment_t1_from_state((*C.struct_whisper_state)(state), C.int(segment)))
}

// Get the text of the specified segment.
func (state *State) Whisper_full_get_segment_text(segment int) string {
	return C.GoString(C.whisper_full_get_segment_text_from_state((*C.struct_whisper_state)(state), C.int(segment)))
}

// Get number of tokens in the specified segment.
func (state *State) Whisper_full_n_tokens(segment int) int {
	return int(C.whisper_full_n_tokens_from_state((*C.struct_whisper_state)(state), C.int(segment)))
}

// Get the token text of the specified token index in the specified segment.
func (state *State) Whisper_full_get_token_text(ctx* Context, segment int, token int) string {
	return C.GoString(C.whisper_full_get_token_text_from_state((*C.struct_whisper_context)(ctx), (*C.struct_whisper_state)(state), C.int(segment), C.int(token)))
}

// Get the token of the specified token index in the specified segment.
func (state *State) Whisper_full_get_token_id(segment int, token int) Token {
	return Token(C.whisper_full_get_token_id_from_state((*C.struct_whisper_state)(state), C.int(segment), C.int(token)))
}

// Get token data for the specified token in the specified segment.
// This contains probabilities, timestamps, etc.
func (state *State) Whisper_full_get_token_data(segment int, token int) TokenData {
	return TokenData(C.whisper_full_get_token_data_from_state((*C.struct_whisper_state)(state), C.int(segment), C.int(token)))
}

// Get the probability of the specified token in the specified segment.
func (state *State) Whisper_full_get_token_p(segment int, token int) float32 {
	return float32(C.whisper_full_get_token_p_from_state((*C.struct_whisper_state)(state), C.int(segment), C.int(token)))
}

///////////////////////////////////////////////////////////////////////////////
// CALLBACKS
func (t TokenData) T0() int64 {
	return int64(t.t0)
}

func (t TokenData) T1() int64 {
	return int64(t.t1)
}

func (t TokenData) Id() Token {
	return Token(t.id)
}
