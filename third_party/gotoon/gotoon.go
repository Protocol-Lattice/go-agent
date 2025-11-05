package gotoon

import (
	"encoding"
	"encoding/base64"
	stdjson "encoding/json"
	"fmt"
	"io"
	"reflect"
	"sort"
	"strconv"
	"strings"
)

type RawMessage = stdjson.RawMessage

type Number = stdjson.Number

type Token = stdjson.Token

type Delim = stdjson.Delim

type Encoder = stdjson.Encoder

type Decoder = stdjson.Decoder

func Marshal(v any) ([]byte, error) {
	return stdjson.Marshal(v)
}

func Unmarshal(data []byte, v any) error {
	return stdjson.Unmarshal(data, v)
}

func MarshalIndent(v any, prefix, indent string) ([]byte, error) {
	return stdjson.MarshalIndent(v, prefix, indent)
}

func NewEncoder(w io.Writer) *Encoder {
	return stdjson.NewEncoder(w)
}

func NewDecoder(r io.Reader) *Decoder {
	return stdjson.NewDecoder(r)
}

func Valid(data []byte) bool {
	return stdjson.Valid(data)
}

type EncodeOption func(*encodeConfig)

type encodeConfig struct {
	indent   string
	sortKeys bool
}

var defaultEncodeConfig = encodeConfig{
	indent:   "  ",
	sortKeys: true,
}

func WithIndent(indent string) EncodeOption {
	return func(cfg *encodeConfig) {
		cfg.indent = indent
	}
}

func WithSortedKeys(sorted bool) EncodeOption {
	return func(cfg *encodeConfig) {
		cfg.sortKeys = sorted
	}
}

func Encode(input any, opts ...EncodeOption) (string, error) {
	cfg := defaultEncodeConfig
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}

	var builder strings.Builder
	if err := encodeValue(&builder, reflect.ValueOf(input), 0, &cfg, false); err != nil {
		return "", err
	}
	return builder.String(), nil
}

func encodeValue(builder *strings.Builder, value reflect.Value, depth int, cfg *encodeConfig, startOnNewLine bool) error {
	if !value.IsValid() {
		if startOnNewLine {
			writeIndent(builder, depth, cfg.indent)
		}
		builder.WriteString("null")
		return nil
	}

	if ok, err := tryMarshalInterfaces(builder, value, depth, cfg, startOnNewLine); ok || err != nil {
		return err
	}

	for value.Kind() == reflect.Interface || value.Kind() == reflect.Pointer {
		if value.IsNil() {
			if startOnNewLine {
				writeIndent(builder, depth, cfg.indent)
			}
			builder.WriteString("null")
			return nil
		}
		value = value.Elem()
	}

	if ok, err := tryMarshalInterfaces(builder, value, depth, cfg, startOnNewLine); ok || err != nil {
		return err
	}

	switch value.Kind() {
	case reflect.Bool:
		if startOnNewLine {
			writeIndent(builder, depth, cfg.indent)
		}
		builder.WriteString(strconv.FormatBool(value.Bool()))
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		if startOnNewLine {
			writeIndent(builder, depth, cfg.indent)
		}
		builder.WriteString(strconv.FormatInt(value.Int(), 10))
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		if startOnNewLine {
			writeIndent(builder, depth, cfg.indent)
		}
		builder.WriteString(strconv.FormatUint(value.Uint(), 10))
	case reflect.Float32:
		if startOnNewLine {
			writeIndent(builder, depth, cfg.indent)
		}
		builder.WriteString(strconv.FormatFloat(value.Float(), 'f', -1, 32))
	case reflect.Float64:
		if startOnNewLine {
			writeIndent(builder, depth, cfg.indent)
		}
		builder.WriteString(strconv.FormatFloat(value.Float(), 'f', -1, 64))
	case reflect.String:
		if startOnNewLine {
			writeIndent(builder, depth, cfg.indent)
		}
		builder.WriteString(strconv.Quote(value.String()))
	case reflect.Map:
		return encodeObject(builder, mapEntries(value, cfg), depth, cfg, startOnNewLine)
	case reflect.Struct:
		return encodeObject(builder, structEntries(value, cfg), depth, cfg, startOnNewLine)
	case reflect.Slice, reflect.Array:
		if value.Type().Elem().Kind() == reflect.Uint8 {
			if startOnNewLine {
				writeIndent(builder, depth, cfg.indent)
			}
			builder.WriteString(strconv.Quote(base64.StdEncoding.EncodeToString(value.Bytes())))
			return nil
		}
		return encodeList(builder, value, depth, cfg, startOnNewLine)
	default:
		return fmt.Errorf("gotoon: unsupported kind %s", value.Kind())
	}

	return nil
}

func tryMarshalInterfaces(builder *strings.Builder, value reflect.Value, depth int, cfg *encodeConfig, startOnNewLine bool) (bool, error) {
	if value.IsValid() && value.CanInterface() {
		if marshaler, ok := value.Interface().(stdjson.Marshaler); ok {
			data, err := marshaler.MarshalJSON()
			if err != nil {
				return true, err
			}
			var intermediate any
			if err := stdjson.Unmarshal(data, &intermediate); err != nil {
				return true, err
			}
			return true, encodeValue(builder, reflect.ValueOf(intermediate), depth, cfg, startOnNewLine)
		}
		if textMarshaler, ok := value.Interface().(encoding.TextMarshaler); ok {
			text, err := textMarshaler.MarshalText()
			if err != nil {
				return true, err
			}
			if startOnNewLine {
				writeIndent(builder, depth, cfg.indent)
			}
			builder.WriteString(strconv.Quote(string(text)))
			return true, nil
		}
	}
	return false, nil
}

type objectEntry struct {
	key   string
	value reflect.Value
}

func mapEntries(value reflect.Value, cfg *encodeConfig) []objectEntry {
	entries := make([]objectEntry, 0, value.Len())
	for _, key := range value.MapKeys() {
		entries = append(entries, objectEntry{
			key:   fmt.Sprint(key.Interface()),
			value: value.MapIndex(key),
		})
	}
	if cfg.sortKeys {
		sort.Slice(entries, func(i, j int) bool {
			return entries[i].key < entries[j].key
		})
	}
	return entries
}

func structEntries(value reflect.Value, cfg *encodeConfig) []objectEntry {
	t := value.Type()
	entries := make([]objectEntry, 0, value.NumField())
	for i := 0; i < value.NumField(); i++ {
		field := t.Field(i)
		if field.PkgPath != "" { // unexported
			continue
		}
		name := field.Name
		if tag, ok := field.Tag.Lookup("json"); ok {
			parts := strings.Split(tag, ",")
			if len(parts) > 0 && parts[0] != "" {
				if parts[0] == "-" {
					continue
				}
				name = parts[0]
			}
		}
		entries = append(entries, objectEntry{
			key:   name,
			value: value.Field(i),
		})
	}
	if cfg.sortKeys {
		sort.Slice(entries, func(i, j int) bool {
			return entries[i].key < entries[j].key
		})
	}
	return entries
}

func encodeObject(builder *strings.Builder, entries []objectEntry, depth int, cfg *encodeConfig, startOnNewLine bool) error {
	if startOnNewLine {
		writeIndent(builder, depth, cfg.indent)
	}
	if len(entries) == 0 {
		builder.WriteString("{}")
		return nil
	}

	builder.WriteString("{\n")
	for _, entry := range entries {
		writeIndent(builder, depth+1, cfg.indent)
		builder.WriteString(entry.key)
		builder.WriteString(":")
		if isScalar(entry.value) {
			builder.WriteByte(' ')
			if err := encodeValue(builder, entry.value, depth+1, cfg, false); err != nil {
				return err
			}
			builder.WriteByte('\n')
			continue
		}

		builder.WriteByte('\n')
		if err := encodeValue(builder, entry.value, depth+1, cfg, true); err != nil {
			return err
		}
		builder.WriteByte('\n')
	}
	writeIndent(builder, depth, cfg.indent)
	builder.WriteString("}")
	return nil
}

func encodeList(builder *strings.Builder, value reflect.Value, depth int, cfg *encodeConfig, startOnNewLine bool) error {
	if startOnNewLine {
		writeIndent(builder, depth, cfg.indent)
	}
	length := value.Len()
	if length == 0 {
		builder.WriteString("[]")
		return nil
	}

	builder.WriteString("[\n")
	for i := 0; i < length; i++ {
		item := value.Index(i)
		if isScalar(item) {
			writeIndent(builder, depth+1, cfg.indent)
			if err := encodeValue(builder, item, depth+1, cfg, false); err != nil {
				return err
			}
			builder.WriteByte('\n')
			continue
		}
		if err := encodeValue(builder, item, depth+1, cfg, true); err != nil {
			return err
		}
		builder.WriteByte('\n')
	}
	writeIndent(builder, depth, cfg.indent)
	builder.WriteString("]")
	return nil
}

func isScalar(value reflect.Value) bool {
	for value.Kind() == reflect.Interface || value.Kind() == reflect.Pointer {
		if value.IsNil() {
			return true
		}
		value = value.Elem()
	}

	switch value.Kind() {
	case reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64, reflect.String:
		return true
	case reflect.Slice:
		return value.Type().Elem().Kind() == reflect.Uint8
	default:
		return false
	}
}

func writeIndent(builder *strings.Builder, depth int, indent string) {
	for i := 0; i < depth; i++ {
		builder.WriteString(indent)
	}
}
