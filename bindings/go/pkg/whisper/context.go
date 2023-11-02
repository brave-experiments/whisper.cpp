package whisper

import (
	"fmt"
	"runtime"
	"strings"
	"time"

	// Bindings
	whisper "github.com/brave-experiments/whisper.cpp/bindings/go"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

type context struct {
	n      int
	model  *model
	params whisper.Params
}

type state struct {
	st *whisper.State
}

func (s *state) Close() error {
	if s.st != nil {
		return s.st.Close()
	}
	return nil
}

// Make sure context adheres to the interface
var _ Context = (*context)(nil)
var _ State = (*state)(nil)

///////////////////////////////////////////////////////////////////////////////
// LIFECYCLE

func newContext(model *model, params whisper.Params) (Context, error) {
	context := new(context)
	context.model = model
	context.params = params

	// Return success
	return context, nil
}

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

func (context *context) NewState() State {
	s := new(state)
	s.st = context.model.ctx.Whisper_init_state()
	return s
}

// Set the language to use for speech recognition.
func (context *context) SetLanguage(lang string) error {
	if context.model.ctx == nil {
		return ErrInternalAppError
	}
	if !context.model.IsMultilingual() {
		return ErrModelNotMultilingual
	}

	if lang == "auto" {
		context.params.SetLanguage(-1)
	} else if id := context.model.ctx.Whisper_lang_id(lang); id < 0 {
		return ErrUnsupportedLanguage
	} else if err := context.params.SetLanguage(id); err != nil {
		return err
	}
	// Return success
	return nil
}

func (context *context) IsMultilingual() bool {
	return context.model.IsMultilingual()
}

// Get language
func (context *context) Language() string {
	id := context.params.Language()
	if id == -1 {
		return "auto"
	}
	return whisper.Whisper_lang_str(context.params.Language())
}

// Set translate flag
func (context *context) SetTranslate(v bool) {
	context.params.SetTranslate(v)
}

// Set speedup flag
func (context *context) SetSpeedup(v bool) {
	context.params.SetSpeedup(v)
}

func (context *context) SetSplitOnWord(v bool) {
	context.params.SetSplitOnWord(v)
}

// Set number of threads to use
func (context *context) SetThreads(v uint) {
	context.params.SetThreads(int(v))
}

// Set time offset
func (context *context) SetOffset(v time.Duration) {
	context.params.SetOffset(int(v.Milliseconds()))
}

// Set duration of audio to process
func (context *context) SetDuration(v time.Duration) {
	context.params.SetDuration(int(v.Milliseconds()))
}

// Set timestamp token probability threshold (~0.01)
func (context *context) SetTokenThreshold(t float32) {
	context.params.SetTokenThreshold(t)
}

// Set timestamp token sum probability threshold (~0.01)
func (context *context) SetTokenSumThreshold(t float32) {
	context.params.SetTokenSumThreshold(t)
}

// Set max segment length in characters
func (context *context) SetMaxSegmentLength(n uint) {
	context.params.SetMaxSegmentLength(int(n))
}

// Set token timestamps flag
func (context *context) SetTokenTimestamps(b bool) {
	context.params.SetTokenTimestamps(b)
}

// Set max tokens per segment (0 = no limit)
func (context *context) SetMaxTokensPerSegment(n uint) {
	context.params.SetMaxTokensPerSegment(int(n))
}
func (context *context) SetSuppressNonSpeechTokens(b bool) {
	context.params.SetSuppressNonSpeechTokens(b)
}

// ResetTimings resets the mode timings. Should be called before processing
func (context *context) ResetTimings() {
	context.model.ctx.Whisper_reset_timings()
}

// PrintTimings prints the model timings to stdout.
func (context *context) PrintTimings() {
	context.model.ctx.Whisper_print_timings()
}

// SystemInfo returns the system information
func (context *context) SystemInfo() string {
	return fmt.Sprintf("system_info: n_threads = %d / %d | %s\n",
		context.params.Threads(),
		runtime.NumCPU(),
		whisper.Whisper_print_system_info(),
	)
}

// Process new sample data and return any errors
func (context *context) Process(
	s State,
	data []float32,
) ([]Segment, error) {
	if context.model.ctx == nil {
		return nil, ErrInternalAppError
	}

	if err := context.model.ctx.Whisper_full_with_state(s.(*state).st, context.params, data); err != nil {
		return nil, err
	}

	num_segments := s.(*state).st.Whisper_full_n_segments()
	segments := make([]Segment, num_segments)
	for i := 0; i < num_segments; i++ {
		segments[i] = toSegment(context.model.ctx, s.(*state).st, i)
	}

	// Return success
	return segments, nil
}

// Test for text tokens
func (context *context) IsText(t Token) bool {
	switch {
	case context.IsBEG(t):
		return false
	case context.IsSOT(t):
		return false
	case whisper.Token(t.Id) >= context.model.ctx.Whisper_token_eot():
		return false
	case context.IsPREV(t):
		return false
	case context.IsSOLM(t):
		return false
	case context.IsNOT(t):
		return false
	default:
		return true
	}
}

// Test for "begin" token
func (context *context) IsBEG(t Token) bool {
	return whisper.Token(t.Id) == context.model.ctx.Whisper_token_beg()
}

// Test for "start of transcription" token
func (context *context) IsSOT(t Token) bool {
	return whisper.Token(t.Id) == context.model.ctx.Whisper_token_sot()
}

// Test for "end of transcription" token
func (context *context) IsEOT(t Token) bool {
	return whisper.Token(t.Id) == context.model.ctx.Whisper_token_eot()
}

// Test for "start of prev" token
func (context *context) IsPREV(t Token) bool {
	return whisper.Token(t.Id) == context.model.ctx.Whisper_token_prev()
}

// Test for "start of lm" token
func (context *context) IsSOLM(t Token) bool {
	return whisper.Token(t.Id) == context.model.ctx.Whisper_token_solm()
}

// Test for "No timestamps" token
func (context *context) IsNOT(t Token) bool {
	return whisper.Token(t.Id) == context.model.ctx.Whisper_token_not()
}

// Test for token associated with a specific language
func (context *context) IsLANG(t Token, lang string) bool {
	if id := context.model.ctx.Whisper_lang_id(lang); id >= 0 {
		return whisper.Token(t.Id) == context.model.ctx.Whisper_token_lang(id)
	} else {
		return false
	}
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE METHODS

func toSegment(ctx *whisper.Context, state *whisper.State, n int) Segment {
	return Segment{
		Num:    n,
		Text:   strings.TrimSpace(state.Whisper_full_get_segment_text(n)),
		Start:  time.Duration(state.Whisper_full_get_segment_t0(n)) * time.Millisecond * 10,
		End:    time.Duration(state.Whisper_full_get_segment_t1(n)) * time.Millisecond * 10,
		Tokens: toTokens(ctx, state, n),
	}
}

func toTokens(ctx *whisper.Context, state *whisper.State, n int) []Token {
	result := make([]Token, state.Whisper_full_n_tokens(n))
	for i := 0; i < len(result); i++ {
		data := state.Whisper_full_get_token_data(n, i)

		result[i] = Token{
			Id:    int(state.Whisper_full_get_token_id(n, i)),
			Text:  state.Whisper_full_get_token_text(ctx, n, i),
			P:     state.Whisper_full_get_token_p(n, i),
			Start: time.Duration(data.T0()) * time.Millisecond * 10,
			End:   time.Duration(data.T1()) * time.Millisecond * 10,
		}
	}
	return result
}
