#ifndef ZENO_BYTEBUFFER_H
#define ZENO_BYTEBUFFER_H

#include <array>
#include <cstdint>
#include <mutex>
#include <algorithm>
#include <memory>
#include <type_traits>
#include <cstring>
#include <cassert>
#include <string>
#include <sstream>
#include <iomanip>
#include "byteorder.h"
#include "networktypes.h"

template <size_t Size>
struct ByteBuffer {

    using ValueType = uint8_t;

    struct Proxy {
        friend struct ByteBuffer;

        Proxy& operator=(ValueType value) {
            std::scoped_lock lock{proxyObject.m_mutex};
            proxyObject[index] = value;
            return *this;
        }

        operator ValueType() const {
            return proxyObject[index];
        }

        private:
            Proxy(ByteBuffer& inObject, size_t inIndex) : proxyObject(inObject), index(inIndex) {};

            ByteBuffer& proxyObject;
            size_t index;
    };

    /** get buffer size */
    constexpr size_t size() const  {
        return Size;
    }

    /** how many bytes data in buffer */
    size_t dataSize() const {
        return m_cursor;
    }

    /**
     * Move cursor by offset value
     * @param value offset value, could be negative
     * @return is success
     */
    bool moveCursor(int32_t value) {
        if ((int64_t)value + m_cursor >= size()) return false;

        std::scoped_lock lock{m_mutex};
        m_cursor += value;
        return true;
    }

    /**
     * Get size of next complete packet.
     * Returning size is included header and split chars.
     * @return size of next complete packet. -1 if not exist.
     */
    int16_t getNextPacketSize() {
        std::scoped_lock lock{m_mutex};

        return  getNextPacketSize_Unsafe();
    }

    int16_t getFirstPacketStartIndex_Unsafe() {
        if (0 == m_cursor) return -1;

        const auto bufEndIter = m_buf.begin() + m_cursor + 1;

        if (zeno::byteorder::is_big_endian()) {
            const auto itPacketStart = std::search(
                m_buf.begin(), bufEndIter,
                g_packetStart.rbegin(), g_packetStart.rend()
            );

            if (itPacketStart != bufEndIter) {
                return std::distance(m_buf.begin(), itPacketStart);
            }
        } else {
            const auto itPacketStart = std::search(
                m_buf.begin(), bufEndIter,
                g_packetStart.begin(), g_packetStart.end()
            );

            if (itPacketStart != bufEndIter) {
                return std::distance(m_buf.begin(), itPacketStart);
            }
        }

        return -1;
    }

    int16_t getFirstPacketEndIndex_Unsafe() {
        if (0 == m_cursor) return -1;

        const auto bufEndIter = m_buf.begin() + m_cursor + 1;

        if (zeno::byteorder::is_big_endian()) {
            const auto itPacketEnd = std::search(
                m_buf.begin(), bufEndIter,
                g_packetSplit.rbegin(), g_packetSplit.rend()
            );

            if (itPacketEnd != bufEndIter) {
                return std::distance(m_buf.begin(), itPacketEnd) + g_packetSplit.size();
            }
        } else {
            const auto itPacketEnd = std::search(
                m_buf.begin(), bufEndIter,
                g_packetSplit.begin(), g_packetSplit.end()
            );

            if (itPacketEnd != bufEndIter) {
                return std::distance(m_buf.begin(), itPacketEnd) + g_packetSplit.size();
            }
        }

        return -1;
    }

    int16_t getNextPacketSize_Unsafe() {
        if (0 == m_cursor) return -1;

        const auto bufEndIter = m_buf.begin() + m_cursor + 1;

        if (zeno::byteorder::is_big_endian()) {
            const auto itPacketStart = std::search(
                m_buf.begin(), bufEndIter,
                g_packetStart.rbegin(), g_packetStart.rend()
            );
            const auto itPacketEnd = std::search(
                m_buf.begin(), bufEndIter,
                g_packetSplit.rbegin(), g_packetSplit.rend()
            );

            if (itPacketEnd != bufEndIter && itPacketStart != bufEndIter) {
                const uint16_t startIndex = std::distance(m_buf.begin(), itPacketStart);
                const int16_t endIndex = std::distance(m_buf.begin(), itPacketEnd) + g_packetSplit.size();
                return endIndex - startIndex;
            }
        } else {
            const auto itPacketStart = std::search(
                m_buf.begin(), bufEndIter,
                g_packetStart.begin(), g_packetStart.end()
            );
            const auto itPacketEnd = std::search(
                m_buf.begin(), bufEndIter,
                g_packetSplit.begin(), g_packetSplit.end()
            );

            if (itPacketEnd != bufEndIter && itPacketStart != bufEndIter) {
                const uint16_t startIndex = std::distance(m_buf.begin(), itPacketStart);
                const int16_t endIndex = std::distance(m_buf.begin(), itPacketEnd) + g_packetSplit.size();
                return endIndex - startIndex;
            }
        }

        return  -1;
    }

    /**
     * Try to read a packet from buffer.
     * Use getNextPacketSize first.
     * @param outPacketBuf the buffer allocated by caller the store packet
     * @return return false if there is no complete packet received
     */
    bool readSinglePacket(uint8_t* outPacketBuf) {
        if (0 == m_cursor) return false;

        std::scoped_lock lock { m_mutex };

        const int16_t packetSize = getNextPacketSize_Unsafe();
        if (packetSize == -1) {
            return false;
        }

        const int16_t packetStartIdx = getFirstPacketStartIndex_Unsafe();
        std::memmove(outPacketBuf, m_buf.data() + packetStartIdx, packetSize);
        shiftForward_Unsafe(packetSize);

        return true;
    }

    /**
     * Move content of array forward.
     * Will not set unused bytes to zero.
     * Clear buffer if size > m_cursor.
     * @param size how many bytes at the front of buffer will replaced
     */
    void shiftForward(size_t size) {
        std::scoped_lock lock { m_mutex };

        shiftForward_Unsafe(size);
    }

    void shiftForward_Unsafe(size_t size) {
        if (size > m_cursor) {
            m_cursor = 0;
            return;
        }

        ValueType* rawData = m_buf.data();
        std::memmove(rawData, rawData + size, m_cursor - size + 1);
        m_cursor -= size;
    }

    /** clean up data not belongs to packet */
    void cleanUp() {
        std::scoped_lock lock { m_mutex };

        const int16_t startIdx = getFirstPacketStartIndex_Unsafe();
        if (startIdx != 0) {
            shiftForward_Unsafe(startIdx);
        }
    }

    /** This operation isn't thread safety */
    ValueType* operator*() {
        return m_buf.data();
    }

    Proxy& operator[](size_t idx) const {
        assert(idx < size());
        return Proxy { *this, idx };
    }

private:
    std::array<ValueType, Size> m_buf;
    /** offset to the end of buffer */
    size_t m_cursor = 0;
    /** mutex offers thread safety */
    std::mutex m_mutex;

public:
};

namespace zeno_bridge {
    template <
        typename DataType,
        typename HeaderType = ZBTPacketHeader,
        typename EndType =  std::decay_t<uint8_t[g_packetSplit.size()]>
    >
    std::tuple<HeaderType*, DataType*, EndType> parsePacketType(uint8_t* data) {
        const size_t headerSize = sizeof(HeaderType), dataSize = sizeof(DataType);
        auto* pHeader = reinterpret_cast<HeaderType*>(data);
        auto* pData = reinterpret_cast<DataType*>(data + headerSize);
        auto* pEnd = reinterpret_cast<EndType>(pData + dataSize);

        return std::make_tuple(pHeader, pData, pEnd);
    }
}

namespace zeno {
template<typename TInputIter>
std::string make_hex_string(TInputIter first, TInputIter last, bool use_uppercase = true, bool insert_spaces = false)
{
    std::ostringstream ss;
    ss << std::hex << std::setfill('0');
    if (use_uppercase)
        ss << std::uppercase;
    while (first != last)
    {
        ss << std::setw(2) << static_cast<int>(*first++);
        if (insert_spaces && first != last)
            ss << " ";
    }
    return ss.str();
}
}

#endif //ZENO_BYTEBUFFER_H
