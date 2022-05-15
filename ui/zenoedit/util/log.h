#ifndef __ZENOEDIT_LOG_H__
#define __ZENOEDIT_LOG_H__

#include <zeno/utils/log.h> 
#include <zeno/utils/zeno_p.h> 

#if !defined(QT_NO_DEBUG)
#define ZASSERT_EXIT(cond, ...) \
    if (!(cond))                \
    {                           \
        Q_ASSERT(cond);         \
        return __VA_ARGS__;     \
    }
#else
#define ZASSERT_EXIT(cond, ...) \
    if (!(cond))                \
    {                           \
        zeno::log_error(#cond); \
        return __VA_ARGS__;     \
    }
#endif

#endif
