#pragma once

// jiao wo cihou da shi
#include <QString>
#include <map>
#include <unordered_map>

#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
#define QtSkipEmptyParts Qt::SkipEmptyParts
#define qt_unordered_map std::unordered_map
#else
#define QtSkipEmptyParts QString::SkipEmptyParts
#define qt_unordered_map std::map
#endif