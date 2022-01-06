#include "uihelper.h"
#include <QUuid>


NODE_DESCS UiHelper::loadDescsFromTempFile()
{
    QFile file(":/templates/tmp-descs.json");
    bool ret = file.open(QIODevice::ReadOnly | QIODevice::Text);
    Q_ASSERT(ret);
    rapidjson::Document doc;
    QByteArray bytes = file.readAll();
    doc.Parse(bytes);
    return UiHelper::parseDescs(doc);
}

NODE_DESCS UiHelper::parseDescs(const rapidjson::Value &jsonDescs)
{
    NODE_DESCS _descs;
    for (const auto &node : jsonDescs.GetObject())
    {
        const QString &name = node.name.GetString();
        if (name == "MakeHeatmap") {
            int j;
            j = 0;
        }
        const auto &objValue = node.value;
        auto inputs = objValue["inputs"].GetArray();
        auto outputs = objValue["outputs"].GetArray();
        auto params = objValue["params"].GetArray();
        auto categories = objValue["categories"].GetArray();

        NODE_DESC desc;

        for (int i = 0; i < inputs.Size(); i++) {
            if (inputs[i].IsArray()) {
                auto input_triple = inputs[i].GetArray();
                const QString &socketType = input_triple[0].GetString();
                const QString &socketName = input_triple[1].GetString();
                const QString &socketDefl = input_triple[2].GetString();
                PARAM_CONTROL ctrlType = _getControlType(socketType);
                INPUT_SOCKET inputSocket;
                inputSocket.info = SOCKET_INFO("", socketName, QPointF(), true);
                inputSocket.info.type = socketType;
                inputSocket.info.control = _getControlType(socketType);
                inputSocket.info.defaultValue = _parseDefaultValue(socketDefl);
                desc.inputs.insert(socketName, inputSocket);
            } else {
            }
        }

        for (int i = 0; i < params.Size(); i++) {
            if (params[i].IsArray()) {
                auto param_triple = params[i].GetArray();
                const QString &socketType = param_triple[0].GetString();
                const QString &socketName = param_triple[1].GetString();
                const QString &socketDefl = param_triple[2].GetString();
                PARAM_CONTROL ctrlType = _getControlType(socketType);
                PARAM_INFO paramInfo;
                paramInfo.bEnableConnect = false;
                paramInfo.control = ctrlType;
                paramInfo.name = socketName;
                paramInfo.typeDesc = socketType;
                paramInfo.defaultValue = _parseDefaultValue(socketDefl);

                desc.params.insert(socketName, paramInfo);
            }
        }

        for (int i = 0; i < outputs.Size(); i++) {
            if (outputs[i].IsArray()) {
                auto output_triple = outputs[i].GetArray();
                const QString &socketType = output_triple[0].GetString();
                const QString &socketName = output_triple[1].GetString();
                const QString &socketDefl = output_triple[2].GetString();
                PARAM_CONTROL ctrlType = _getControlType(socketType);
                OUTPUT_SOCKET outputSocket;
                outputSocket.info = SOCKET_INFO("", socketName, QPointF(), false);
                outputSocket.info.type = socketType;
                outputSocket.info.control = _getControlType(socketType);
                outputSocket.info.defaultValue = _parseDefaultValue(socketDefl);

                desc.outputs.insert(socketName, outputSocket);
            } else {
            }
        }
        
        for (int i = 0; i < categories.Size(); i++)
        {
            desc.categories.push_back(categories[i].GetString());
        }

        _descs.insert(name, desc);
    }
    return _descs;
}

QVariant UiHelper::_parseDefaultValue(const QString &defaultValue)
{
    //some data like vec3f, cast to string first.
    bool bOk = false;
    float val = defaultValue.toFloat(&bOk);
    QVariant var;
    if (bOk) {
        var = val;
    } else {
        var = defaultValue;
    }
    return var;
}

QString UiHelper::generateUuid(const QString& name)
{
    QUuid uuid = QUuid::createUuid();
    return QString::number(uuid.data1, 16) + "-" + name;
}

PARAM_CONTROL UiHelper::_getControlType(const QString &type)
{
    if (type == "int") {
        return CONTROL_INT;
    } else if (type == "bool") {
        return CONTROL_BOOL;
    } else if (type == "float") {
        return CONTROL_FLOAT;
    } else if (type == "string") {
        return CONTROL_STRING;
    } else if (type == "writepath") {
        return CONTROL_WRITEPATH;
    } else if (type == "readpath") {
        return CONTROL_READPATH;
    } else if (type == "multiline_string") {
        return CONTROL_MULTILINE_STRING;
    } else if (type == "_RAMPS") {
        return CONTROL_HEAPMAP;
    } else if (type.startsWith("enum ")) {
        return CONTROL_ENUM;
    } else {
        return CONTROL_NONE;
    }
}

std::pair<qreal, qreal> UiHelper::getRxx2(QRectF r, qreal xRadius, qreal yRadius, bool AbsoluteSize)
{
    if (AbsoluteSize) {
        qreal w = r.width() / 2;
        qreal h = r.height() / 2;

        if (w == 0) {
            xRadius = 0;
        } else {
            xRadius = 100 * qMin(xRadius, w) / w;
        }
        if (h == 0) {
            yRadius = 0;
        } else {
            yRadius = 100 * qMin(yRadius, h) / h;
        }
    } else {
        if (xRadius > 100)// fix ranges
            xRadius = 100;

        if (yRadius > 100)
            yRadius = 100;
    }

    qreal w = r.width();
    qreal h = r.height();
    qreal rxx2 = w * xRadius / 100;
    qreal ryy2 = h * yRadius / 100;
    return std::make_pair(rxx2, ryy2);
}

QPainterPath UiHelper::getRoundPath(QRectF r, int lt_radius, int rt_radius, int lb_radius, int rb_radius, bool bFixRadius) {
    QPainterPath path;
    if (r.isNull())
        return path;

    if (lt_radius <= 0 && rt_radius <= 0 && lb_radius <= 0 && rb_radius <= 0) {
        path.addRect(r);
        return path;
    }

    qreal x = r.x();
    qreal y = r.y();
    qreal w = r.width();
    qreal h = r.height();

    auto pair = getRxx2(r, lt_radius, lt_radius, bFixRadius);
    qreal rxx2 = pair.first, ryy2 = pair.second;
    if (rxx2 <= 0) {
        path.moveTo(x, y);
    } else {
        path.arcMoveTo(x, y, rxx2, ryy2, 180);
        path.arcTo(x, y, rxx2, ryy2, 180, -90);
    }

    pair = getRxx2(r, rt_radius, rt_radius, bFixRadius);
    rxx2 = pair.first, ryy2 = pair.second;
    if (rxx2 <= 0) {
        path.lineTo(x + w, y);
    } else {
        path.arcTo(x + w - rxx2, y, rxx2, ryy2, 90, -90);
    }

    pair = getRxx2(r, rb_radius, rb_radius, bFixRadius);
    rxx2 = pair.first, ryy2 = pair.second;
    if (rxx2 <= 0) {
        path.lineTo(x + w, y + h);
    } else {
        path.arcTo(x + w - rxx2, y + h - rxx2, rxx2, ryy2, 0, -90);
    }

    pair = getRxx2(r, lb_radius, lb_radius, bFixRadius);
    rxx2 = pair.first, ryy2 = pair.second;
    if (rxx2 <= 0) {
        path.lineTo(x, y + h);
    } else {
        path.arcTo(x, y + h - rxx2, rxx2, ryy2, 270, -90);
    }

    path.closeSubpath();
    return path;
}