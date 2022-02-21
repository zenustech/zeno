#ifndef __UI_HELPER_H__
#define __UI_HELPER_H__

#include "../model/modeldata.h"
#include <rapidjson/document.h>

class UiHelper
{
public:
    static NODE_DESCS parseDescs(const rapidjson::Value &descs);
    static QPainterPath getRoundPath(QRectF r, int lt, int rt, int lb, int rb, bool bFixRadius);
    static QString generateUuid(const QString &name = "x");
    static QVariant _parseDefaultValue(const QString& defaultValue, const QString &type);
    static QVariant parseVariantValue(const rapidjson::Value& val);
    static QVariant parseTextValue(PARAM_CONTROL editCtrl, const QString& textValue);
    static QSizeF viewItemTextLayout(QTextLayout& textLayout, int lineWidth, int maxHeight = -1, int* lastVisibleLine = nullptr);
    static PARAM_CONTROL _getControlType(const QString& type);
    static QString variantToString(const QVariant& var);
private:
    static std::pair<qreal, qreal> getRxx2(QRectF r, qreal xRadius, qreal yRadius, bool AbsoluteSize);
};

#endif
