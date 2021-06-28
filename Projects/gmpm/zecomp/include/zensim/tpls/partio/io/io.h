#pragma once
#include "ZIP.h"

#include <fstream>
#include <ios>
#include <iosfwd>
#include <string>
#include <memory>

namespace Partio {
namespace io {

template <typename T>
inline void make_stream_locale_independent(T& stream)
{
    if (stream) {
        stream->imbue(std::locale::classic());
    }
}

inline std::unique_ptr<std::istream> unzip(const std::string& filename)
{
    std::unique_ptr<std::istream> input(Gzip_In(filename, std::ios::in));
    make_stream_locale_independent(input);
    return input;
}

inline std::unique_ptr<std::istream> read(const std::string& filename)
{
    std::unique_ptr<std::istream> input(
        new std::ifstream(filename, std::ios::in | std::ios::binary));
    make_stream_locale_independent(input);
    return input;
}

inline std::unique_ptr<std::ostream> write(const std::string& filename, bool compressed)
{
    std::unique_ptr<std::ostream> output(
        compressed
        ? Gzip_Out(filename, std::ios::out | std::ios::binary)
        : new std::ofstream(filename, std::ios::out | std::ios::binary));
    make_stream_locale_independent(output);
    return output;
}

inline std::unique_ptr<std::ostream> write(const std::string& filename)
{
    std::unique_ptr<std::ostream> output(
        new std::ofstream(filename, std::ios::out | std::ios::binary));
    make_stream_locale_independent(output);
    return output;
}

}  // io
}  // Partio
