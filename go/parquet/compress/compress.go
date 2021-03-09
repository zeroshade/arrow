package compress

import (
	"compress/flate"
	"io"
	"io/ioutil"

	"github.com/apache/arrow/go/parquet/internal/gen-go/parquet"
)

type Compression parquet.CompressionCodec

func (c Compression) String() string {
	return parquet.CompressionCodec(c).String()
}

const DefaultCompressionLevel = flate.DefaultCompression

var Codecs = struct {
	Uncompressed Compression
	Snappy       Compression
	Gzip         Compression
	Lzo          Compression
	Brotli       Compression
	Lz4          Compression
	Zstd         Compression
}{
	Uncompressed: Compression(parquet.CompressionCodec_UNCOMPRESSED),
	Snappy:       Compression(parquet.CompressionCodec_SNAPPY),
	Gzip:         Compression(parquet.CompressionCodec_GZIP),
	Lzo:          Compression(parquet.CompressionCodec_LZO),
	Brotli:       Compression(parquet.CompressionCodec_BROTLI),
	Lz4:          Compression(parquet.CompressionCodec_LZ4),
	Zstd:         Compression(parquet.CompressionCodec_ZSTD),
}

type Codec interface {
	NewReader(io.Reader) io.ReadCloser
	NewWriter(io.Writer) io.WriteCloser
	NewWriterLevel(io.Writer, int) (io.WriteCloser, error)
	Encode(dst, src []byte) []byte
	EncodeLevel(dst, src []byte, level int) []byte
	CompressBound(int64) int64
	Decode(dst, src []byte) []byte
}

var codecs = map[Compression]Codec{}

type nocodec struct{}

func (nocodec) NewReader(r io.Reader) io.ReadCloser {
	ret, ok := r.(io.ReadCloser)
	if !ok {
		return ioutil.NopCloser(r)
	}
	return ret
}

func (nocodec) Decode(dst, src []byte) []byte {
	if dst != nil {
		copy(dst, src)
	}
	return dst
}

type writerNopCloser struct {
	io.Writer
}

func (writerNopCloser) Close() error {
	return nil
}

func (nocodec) Encode(dst, src []byte) []byte {
	copy(dst, src)
	return dst
}

func (nocodec) EncodeLevel(dst, src []byte, _ int) []byte {
	copy(dst, src)
	return dst
}

func (nocodec) NewWriter(w io.Writer) io.WriteCloser {
	ret, ok := w.(io.WriteCloser)
	if !ok {
		return writerNopCloser{w}
	}
	return ret
}

func (n nocodec) NewWriterLevel(w io.Writer, _ int) (io.WriteCloser, error) {
	return n.NewWriter(w), nil
}

func (nocodec) CompressBound(len int64) int64 { return len }

func init() {
	codecs[Codecs.Uncompressed] = nocodec{}
}

func GetCodec(typ Compression) Codec {
	ret, ok := codecs[typ]
	if !ok {
		// return codecs[Codecs.Uncompressed]
		panic("compression for " + typ.String() + " unimplemented")
	}
	return ret
}
