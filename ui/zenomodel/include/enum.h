#ifndef __ZENOMODEL_ENUM_H__
#define __ZENOMODEL_ENUM_H__

#include <memory>
#include <string>
#include <variant>
#include <vector>
#include <zeno/utils/vec.h>

typedef std::variant<std::string, int, float, double, bool,
    zeno::vec2i, zeno::vec2f, zeno::vec3i, zeno::vec3f, zeno::vec4i, zeno::vec4f> ZVARIANT;

typedef uint32_t ZENO_HANDLE;
typedef uint32_t ZENO_ERROR;

enum SocketType
{
    ST_NONE,
    ST_INT,
    ST_BOOL,
    ST_FLOAT,
    ST_STRING,
    ST_VEC3F,
    ST_VEC3I,
    ST_VEC4F,
    ST_CURVE,
    ST_COLOR,
};

enum ErrorCode
{
    Err_NoError,
    Err_ModelNull,
    Err_IOError,
    Err_NodeNotExist,
    Err_SockNotExist,
    Err_ParamNotFound,
    Err_NoConnection,
    Err_SubgNotExist,
    Err_NotImpl
};


#endif