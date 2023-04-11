#include "zeno/unreal/ZenoUnrealTypes.h"
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/prettywriter.h>
#include <variant>

using namespace rapidjson;

zeno::unreal::GenericFieldVisitor::GenericFieldVisitor(MemoryPoolAllocator<CrtAllocator>& InAllocator)
    : allocator(InAllocator)
{}

zeno::SimpleCharBuffer::SimpleCharBuffer(const char *InChar, size_t Size) {
    length = Size + 1;
    data = new char[length];
    memcpy(data, InChar, Size * sizeof(char));
    data[Size] = '\0';
}

zeno::SimpleCharBuffer::SimpleCharBuffer(zeno::SimpleCharBuffer &&InBuffer) noexcept {
    length = InBuffer.length;
    data = InBuffer.data;
    InBuffer.data = nullptr;
    InBuffer.length = 0;
}

zeno::SimpleCharBuffer &zeno::SimpleCharBuffer::operator=(zeno::SimpleCharBuffer &&InBuffer) noexcept {
    length = InBuffer.length;
    data = InBuffer.data;
    InBuffer.data = nullptr;
    InBuffer.length = 0;
    return *this;
}

zeno::SimpleCharBuffer::~SimpleCharBuffer() {
    delete []data;
}

zeno::SimpleCharBuffer::SimpleCharBuffer(const char *InChar) : length(0)
{
    length = std::strlen(InChar);
    data = new char[length];
    memcpy(data, InChar, length * sizeof(char));
}

zeno::unreal::Mesh::Mesh(const std::vector<zeno::vec3f> &verts, const std::vector<zeno::vec3i> &trigs) {
    vertices.reserve(verts.size());
    for (const auto &item : verts) {
        vertices.push_back(item);
    }
    triangles.reserve(trigs.size());
    for (const auto &item : trigs) {
        triangles.push_back(item);
    }
}
