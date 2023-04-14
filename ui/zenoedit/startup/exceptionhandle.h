#ifndef __EXCEPTION_HANDLER_H__
#define __EXCEPTION_HANDLER_H__

#include <QtWidgets>

#if defined(Q_OS_WIN)
void registerExceptionFilter();
#endif

#endif