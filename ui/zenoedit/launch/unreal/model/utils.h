#ifndef ZENO_UTILS_H
#define ZENO_UTILS_H

#include <type_traits>
#include <zeno/utils/log.h>

/**
 * Allocating new memory block with sizeof(T) and copy item into it.
 * This function isn't considering endian yet.
 * @tparam T any pod type
 * @param item item need to be copy
 * @return Returning pointer to allocated memory, it contains a copy of item.
 * @return Return nullptr if error like sizeof(T) == 0 occurs.
 */
template<typename T, typename = typename std::enable_if<std::is_pod<T>::value>::type>
char* toBytes(const T& item) {
    const size_t sizeOfItem = sizeof(T);
    if (sizeOfItem == 0) return nullptr;

    char* memBlock = static_cast<char *>(malloc(sizeOfItem));
    if (nullptr == memBlock) {
        zeno::log_error("Failed to alloc memory for {}. Sizeof it is {} byte(s)",  typeid(T).name(), sizeOfItem);
        return nullptr;
    }

    std::memmove(memBlock, &item, sizeOfItem);

    return memBlock;
}

#endif //ZENO_UTILS_H
