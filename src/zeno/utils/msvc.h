#pragma once

// jiao wo cihou da shi

#ifdef QT_VERSION
#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
#define QtSkipEmptyParts Qt::SkipEmptyParts
#define msc_unordered_map std::map
#else
#define QtSkipEmptyParts QString::SkipEmptyParts
#endif
#endif