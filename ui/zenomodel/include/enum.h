#ifndef __ZENOMODEL_ENUM_H__
#define __ZENOMODEL_ENUM_H__

#include <memory>
#include <string>
#include <variant>
#include <vector>

typedef std::variant<std::string, int, float, double, bool> ZVARIANT;
typedef uint32_t ZENO_HANDLE;
typedef uint32_t ZENO_ERROR;

enum NodeStatus
{
    STATUS_VIEW,
    STATUS_MUTE,
    STATUS_ONCE,
};

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
};


#endif