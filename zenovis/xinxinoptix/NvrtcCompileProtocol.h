#pragma once

#include <cstdint>
#include <fstream>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace zeno_nvrtc_ipc {

inline constexpr uint32_t RequestMagic = 0x52564e5au;
inline constexpr uint32_t ResponseMagic = 0x52434e5au;
inline constexpr uint32_t ProtocolVersion = 1u;

struct Request {
    std::string source;
    std::string name;
    std::vector<std::string> options;
};

struct Response {
    bool success = false;
    std::string data;
    std::string log;
};

inline void writeU32(std::ostream& os, uint32_t value)
{
    os.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

inline uint32_t readU32(std::istream& is)
{
    uint32_t value = 0;
    is.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!is) {
        throw std::runtime_error("failed to read uint32");
    }
    return value;
}

inline void writeString(std::ostream& os, const std::string& value)
{
    writeU32(os, static_cast<uint32_t>(value.size()));
    os.write(value.data(), static_cast<std::streamsize>(value.size()));
}

inline std::string readString(std::istream& is)
{
    const auto size = readU32(is);
    std::string value(size, '\0');
    if (size != 0) {
        is.read(value.data(), static_cast<std::streamsize>(size));
        if (!is) {
            throw std::runtime_error("failed to read string");
        }
    }
    return value;
}

inline void writeRequest(std::ostream& os, const Request& request)
{
    writeU32(os, RequestMagic);
    writeU32(os, ProtocolVersion);
    writeString(os, request.source);
    writeString(os, request.name);
    writeU32(os, static_cast<uint32_t>(request.options.size()));
    for (const auto& option : request.options) {
        writeString(os, option);
    }
}

inline void writeRequest(const std::string& path, const Request& request)
{
    std::ofstream os(path, std::ios::binary);
    if (!os) {
        throw std::runtime_error("failed to open nvrtc request file");
    }
    writeRequest(os, request);
}

inline Request readRequest(std::istream& is)
{
    if (readU32(is) != RequestMagic || readU32(is) != ProtocolVersion) {
        throw std::runtime_error("bad nvrtc request header");
    }

    Request request;
    request.source = readString(is);
    request.name = readString(is);
    const auto option_count = readU32(is);
    request.options.reserve(option_count);
    for (uint32_t i = 0; i < option_count; ++i) {
        request.options.push_back(readString(is));
    }
    return request;
}

inline Request readRequest(const std::string& path)
{
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        throw std::runtime_error("failed to open nvrtc request file");
    }
    return readRequest(is);
}

inline void writeResponse(std::ostream& os, const Response& response)
{
    writeU32(os, ResponseMagic);
    writeU32(os, ProtocolVersion);
    writeU32(os, response.success ? 1u : 0u);
    writeString(os, response.data);
    writeString(os, response.log);
}

inline void writeResponse(const std::string& path, const Response& response)
{
    std::ofstream os(path, std::ios::binary);
    if (!os) {
        throw std::runtime_error("failed to open nvrtc response file");
    }
    writeResponse(os, response);
}

inline Response readResponse(std::istream& is)
{
    if (readU32(is) != ResponseMagic || readU32(is) != ProtocolVersion) {
        throw std::runtime_error("bad nvrtc response header");
    }

    Response response;
    response.success = readU32(is) != 0;
    response.data = readString(is);
    response.log = readString(is);
    return response;
}

inline Response readResponse(const std::string& path)
{
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        throw std::runtime_error("failed to open nvrtc response file");
    }
    return readResponse(is);
}

} // namespace zeno_nvrtc_ipc
