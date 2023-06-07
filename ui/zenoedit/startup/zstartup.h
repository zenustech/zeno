#ifndef __ZSTARTUP_H__
#define __ZSTARTUP_H__

#include <string>
#include <QCefContext.h>

void startUp();
void verifyVersion();
std::string getZenoVersion();
QCefConfig initCef();

#endif
