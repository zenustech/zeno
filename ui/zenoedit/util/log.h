#ifndef __ZENOEDIT_LOG_H__
#define __ZENOEDIT_LOG_H__

#define ZASSERT_EXIT(cond, ...) \
    Q_ASSERT(cond);\
    if (!(cond))\
        return __VA_ARGS__;

#endif