package whisper

import (
	"io"
	"time"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

type State interface {
	io.Closer
}

// Model is the interface to a whisper model. Create a new model with the
// function whisper.New(string)
type Model interface {
	io.Closer

	// Return a new speech-to-text context.
	NewContext() (Context, error)

	// Return true if the model is multilingual.
	IsMultilingual() bool

	// Return all languages supported.
	Languages() []string
}

// Context is the speach recognition context.
type Context interface {
	NewState() State
	SetLanguage(string) error // Set the language to use for speech recognition, use "auto" for auto detect language.
	SetTranslate(bool)        // Set translate flag
	IsMultilingual() bool     // Return true if the model is multilingual.
	Language() string         // Get language

	SetOffset(time.Duration)      // Set offset
	SetDuration(time.Duration)    // Set duration
	SetThreads(uint)              // Set number of threads to use
	SetSpeedup(bool)              // Set speedup flag
	SetSplitOnWord(bool)          // Set split on word flag
	SetTokenThreshold(float32)    // Set timestamp token probability threshold
	SetTokenSumThreshold(float32) // Set timestamp token sum probability threshold
	SetMaxSegmentLength(uint)     // Set max segment length in characters
	SetTokenTimestamps(bool)      // Set token timestamps flag
	SetMaxTokensPerSegment(uint)  // Set max tokens per segment (0 = no limit)
	SetSuppressNonSpeechTokens(bool)

	// Process mono audio data and return any errors.
	// If defined, newly generated segments are passed to the
	// callback function during processing.
	Process(State, []float32) ([]Segment, error)

	IsBEG(Token) bool          // Test for "begin" token
	IsSOT(Token) bool          // Test for "start of transcription" token
	IsEOT(Token) bool          // Test for "end of transcription" token
	IsPREV(Token) bool         // Test for "start of prev" token
	IsSOLM(Token) bool         // Test for "start of lm" token
	IsNOT(Token) bool          // Test for "No timestamps" token
	IsLANG(Token, string) bool // Test for token associated with a specific language
	IsText(Token) bool         // Test for text token

	// Timings
	PrintTimings()
	ResetTimings()

	SystemInfo() string
}

// Segment is the text result of a speech recognition.
type Segment struct {
	// Segment Number
	Num int

	// Time beginning and end timestamps for the segment.
	Start, End time.Duration

	// The text of the segment.
	Text string

	// The tokens of the segment.
	Tokens []Token
}

// Token is a text or special token
type Token struct {
	Id         int
	Text       string
	P          float32
	Start, End time.Duration
}
