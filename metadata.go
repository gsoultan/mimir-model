package mmf

import (
	"math"
	"sort"
	"unicode/utf8"
)

// Metadata is the typed key-value section. Keys are namespaced with dots
// ("general.name", "tokenizer.vocab"). Values are one of the Go types listed
// in MetaType; arrays are homogeneous.
type Metadata map[string]any

// MetaType identifies the encoded type of a metadata value.
type MetaType uint8

const (
	MetaInt      MetaType = 1 // int64
	MetaUint     MetaType = 2 // uint64
	MetaFloat    MetaType = 3 // float64
	MetaBool     MetaType = 4 // bool
	MetaString   MetaType = 5 // string
	MetaBytes    MetaType = 6 // []byte
	MetaInts     MetaType = 7 // []int64
	MetaFloats   MetaType = 8 // []float64
	MetaStrings  MetaType = 9 // []string
	metaTypeLast          = MetaStrings
)

const (
	metadataMagic     = "MMFM"
	metaHeaderSize    = 16
	metaEntrySize     = 32
	MaxMetadataKeyLen = 1024
	MaxMetadataCount  = 1 << 24
)

// Encode serializes m. Entries are written sorted by key so a reader can
// look a key up by binary search without decoding the whole section.
func (m Metadata) Encode() ([]byte, error) {
	keys := make([]string, 0, len(m))
	for k := range m {
		if len(k) == 0 || len(k) > MaxMetadataKeyLen || !utf8.ValidString(k) {
			return nil, errf(CodeBadMetadata, "key %q", k)
		}
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var pool []byte
	addStr := func(s string) (uint32, uint32) {
		off := uint32(len(pool))
		pool = append(pool, s...)
		return off, uint32(len(s))
	}
	entries := make([]byte, 0, len(keys)*metaEntrySize)
	for _, k := range keys {
		e := make([]byte, metaEntrySize)
		off, n := addStr(k)
		le.PutUint32(e[0:], off)
		le.PutUint32(e[4:], n)
		var t MetaType
		var count uint32
		var payload uint64
		switch v := m[k].(type) {
		case int64:
			t, payload = MetaInt, uint64(v)
		case int:
			t, payload = MetaInt, uint64(int64(v))
		case uint64:
			t, payload = MetaUint, v
		case float64:
			t, payload = MetaFloat, math.Float64bits(v)
		case bool:
			t = MetaBool
			if v {
				payload = 1
			}
		case string:
			if !utf8.ValidString(v) {
				return nil, errf(CodeBadMetadata, "key %q: invalid UTF-8", k)
			}
			o, n := addStr(v)
			t, payload, count = MetaString, uint64(o), n
		case []byte:
			o := uint32(len(pool))
			pool = append(pool, v...)
			t, payload, count = MetaBytes, uint64(o), uint32(len(v))
		case []int64:
			o, _ := alignUp(uint64(len(pool)), 8)
			pool = append(pool, make([]byte, int(o)-len(pool))...)
			for _, x := range v {
				pool = le.AppendUint64(pool, uint64(x))
			}
			t, payload, count = MetaInts, o, uint32(len(v))
		case []float64:
			o, _ := alignUp(uint64(len(pool)), 8)
			pool = append(pool, make([]byte, int(o)-len(pool))...)
			for _, x := range v {
				pool = le.AppendUint64(pool, math.Float64bits(x))
			}
			t, payload, count = MetaFloats, o, uint32(len(v))
		case []string:
			// Table of (offset, length) pairs followed by the strings.
			tab := make([]byte, 8*len(v))
			var body []byte
			base := uint32(len(pool)) + uint32(len(tab))
			for i, s := range v {
				if !utf8.ValidString(s) {
					return nil, errf(CodeBadMetadata, "key %q[%d]: invalid UTF-8", k, i)
				}
				le.PutUint32(tab[8*i:], base+uint32(len(body)))
				le.PutUint32(tab[8*i+4:], uint32(len(s)))
				body = append(body, s...)
			}
			o := uint32(len(pool))
			pool = append(pool, tab...)
			pool = append(pool, body...)
			t, payload, count = MetaStrings, uint64(o), uint32(len(v))
		default:
			return nil, errf(CodeBadMetadata, "key %q: unsupported type %T", k, m[k])
		}
		e[8] = byte(t)
		le.PutUint32(e[12:], count)
		le.PutUint64(e[16:], payload)
		entries = append(entries, e...)
	}
	out := make([]byte, metaHeaderSize, metaHeaderSize+len(entries)+len(pool))
	copy(out, metadataMagic)
	le.PutUint32(out[4:], uint32(len(keys)))
	le.PutUint64(out[8:], uint64(len(pool)))
	out = append(out, entries...)
	return append(out, pool...), nil
}

// DecodeMetadata parses and validates a metadata section.
func DecodeMetadata(b []byte) (Metadata, error) {
	if len(b) < metaHeaderSize || string(b[:4]) != metadataMagic {
		return nil, errf(CodeBadMetadata, "missing header")
	}
	n := le.Uint32(b[4:])
	poolLen := le.Uint64(b[8:])
	if n > MaxMetadataCount || uint64(metaHeaderSize)+uint64(n)*metaEntrySize+poolLen != uint64(len(b)) {
		return nil, errf(CodeBadMetadata, "section length does not match %d entries and %d pool bytes", n, poolLen)
	}
	pool := b[uint64(metaHeaderSize)+uint64(n)*metaEntrySize:]
	str := func(off, ln uint64) (string, error) {
		if !inBounds(off, ln, uint64(len(pool))) {
			return "", errf(CodeBadMetadata, "string outside pool")
		}
		s := pool[off : off+ln]
		if !utf8.Valid(s) {
			return "", errf(CodeBadMetadata, "invalid UTF-8")
		}
		return string(s), nil
	}
	m := make(Metadata, n)
	var prev string
	for i := uint32(0); i < n; i++ {
		e := b[metaHeaderSize+int(i)*metaEntrySize:]
		key, err := str(uint64(le.Uint32(e[0:])), uint64(le.Uint32(e[4:])))
		if err != nil {
			return nil, err
		}
		if key == "" || len(key) > MaxMetadataKeyLen || (i > 0 && key <= prev) {
			return nil, errf(CodeBadMetadata, "key %q out of order or invalid", key)
		}
		prev = key
		t := MetaType(e[8])
		if e[9] != 0 || e[10] != 0 || e[11] != 0 {
			return nil, errf(CodeReservedNonZero, "metadata entry %d", i)
		}
		count := uint64(le.Uint32(e[12:]))
		payload := le.Uint64(e[16:])
		if le.Uint64(e[24:]) != 0 {
			return nil, errf(CodeReservedNonZero, "metadata entry %d", i)
		}
		scalar := t == MetaInt || t == MetaUint || t == MetaFloat || t == MetaBool
		if scalar && count != 0 {
			return nil, errf(CodeBadMetadata, "key %q: scalar with count", key)
		}
		switch t {
		case MetaInt:
			m[key] = int64(payload)
		case MetaUint:
			m[key] = payload
		case MetaFloat:
			m[key] = math.Float64frombits(payload)
		case MetaBool:
			if payload > 1 {
				return nil, errf(CodeBadMetadata, "key %q: bool value %d", key, payload)
			}
			m[key] = payload == 1
		case MetaString:
			s, err := str(payload, count)
			if err != nil {
				return nil, err
			}
			m[key] = s
		case MetaBytes:
			if !inBounds(payload, count, uint64(len(pool))) {
				return nil, errf(CodeBadMetadata, "key %q: bytes outside pool", key)
			}
			m[key] = append([]byte(nil), pool[payload:payload+count]...)
		case MetaInts, MetaFloats:
			if payload%8 != 0 || !inBounds(payload, count*8, uint64(len(pool))) {
				return nil, errf(CodeBadMetadata, "key %q: array outside pool or misaligned", key)
			}
			if t == MetaInts {
				v := make([]int64, count)
				for j := range v {
					v[j] = int64(le.Uint64(pool[payload+uint64(j)*8:]))
				}
				m[key] = v
			} else {
				v := make([]float64, count)
				for j := range v {
					v[j] = math.Float64frombits(le.Uint64(pool[payload+uint64(j)*8:]))
				}
				m[key] = v
			}
		case MetaStrings:
			if !inBounds(payload, count*8, uint64(len(pool))) {
				return nil, errf(CodeBadMetadata, "key %q: string table outside pool", key)
			}
			v := make([]string, count)
			for j := range v {
				tab := pool[payload+uint64(j)*8:]
				s, err := str(uint64(le.Uint32(tab[0:])), uint64(le.Uint32(tab[4:])))
				if err != nil {
					return nil, err
				}
				v[j] = s
			}
			m[key] = v
		default:
			return nil, errf(CodeBadMetadata, "key %q: type %d", key, t)
		}
	}
	return m, nil
}

// Metadata decodes the file's metadata section, or returns an empty map
// when the file has none.
func (f *File) Metadata() (Metadata, error) {
	s, ok := f.Section(SectionMetadata)
	if !ok {
		return Metadata{}, nil
	}
	return DecodeMetadata(f.SectionBytes(s))
}

// Well-known metadata keys.
const (
	KeyName            = "general.name"
	KeyArchitecture    = "general.architecture"
	KeyLicense         = "general.license"
	KeyQuantRecipe     = "general.quantization_recipe"
	KeySplitIndex      = "split.index"
	KeySplitCount      = "split.count"
	KeyTokenizerModel  = "tokenizer.model"
	KeyTokenizerTokens = "tokenizer.tokens"
	KeyTokenizerMerges = "tokenizer.merges"
	KeyTokenizerScores = "tokenizer.scores"
	KeyChatTemplate    = "tokenizer.chat_template"
)
