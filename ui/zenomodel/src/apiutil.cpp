#include "apiutil.h"
#include "modeldata.h"


ZVARIANT ApiUtil::qVarToStdVar(const QVariant& qvar, QString typeDesc)
{
    ZVARIANT var;
    switch (qvar.type())
    {
    case QVariant::Int:
        var = qvar.toInt();
        break;
    case QVariant::Bool:
        var = qvar.toBool();
        break;
    case QMetaType::Float:
        var = qvar.toFloat();
        break;
    case QVariant::Double:
        var = qvar.toDouble();
        break;
    case QVariant::String:
        var = qvar.toString().toStdString();
        break;
    case QVariant::UserType:
        if (qvar.userType() == QMetaTypeId<UI_VECTYPE>::qt_metatype_id())
        {
            UI_VECTYPE qVec = qvar.value<UI_VECTYPE>();
            if (qVec.size() == 2)
            {
                if (typeDesc == "vec2i")
                    var = zeno::vec2i(qVec[0], qVec[1]);
                else if (typeDesc == "vec2f")
                    var = zeno::vec2f(qVec[0], qVec[1]);
            }
            else if (qVec.size() == 3)
            {
                if (typeDesc == "vec3i")
                    var = zeno::vec3i(qVec[0], qVec[1], qVec[2]);
                else if (typeDesc == "vec3f")
                    var = zeno::vec3f(qVec[0], qVec[1], qVec[2]);
            }
            else if (qVec.size() == 4)
            {
                if (typeDesc == "vec4i")
                    var = zeno::vec4f(qVec[0], qVec[1], qVec[2], qVec[3]);
                else if (typeDesc == "vec4f")
                    var = zeno::vec4f(qVec[0], qVec[1], qVec[2], qVec[3]);
            }
        }
        //todo: lineargradient
        break;
    default:
        break;
    };
    return var;
}

QVariant ApiUtil::stdVarToQVar(const ZVARIANT& var)
{
    int idx = var.index();
    QVariant qvar;
    switch (idx)
    {
        case 0:
            qvar = QString::fromStdString(std::get<std::string>(var));
            break;
        case 1:
            qvar = std::get<int>(var);
            break;
        case 2:
            qvar = std::get<float>(var);
            break;
        case 3:
            qvar = std::get<double>(var);
            break;
        case 4:
            qvar = std::get<bool>(var);
            break;
        case 5:
        {
            zeno::vec2i vec = std::get<zeno::vec2i>(var);
            UI_VECTYPE qVec;
            qVec.push_back(vec[0]);
            qVec.push_back(vec[1]);
            qvar = QVariant::fromValue(qVec);
            break;
        }
        case 6:
        {
            zeno::vec2f vec = std::get<zeno::vec2f>(var);
            UI_VECTYPE qVec;
            qVec.push_back(vec[0]);
            qVec.push_back(vec[1]);
            qvar = QVariant::fromValue(qVec);
            break;
        }
        case 7:
        {
            zeno::vec3i vec = std::get<zeno::vec3i>(var);
            UI_VECTYPE qVec;
            qVec.push_back(vec[0]);
            qVec.push_back(vec[1]);
            qVec.push_back(vec[2]);
            qvar = QVariant::fromValue(qVec);
            break;
        }
        case 8:
        {
            zeno::vec3f vec = std::get<zeno::vec3f>(var);
            UI_VECTYPE qVec;
            qVec.push_back(vec[0]);
            qVec.push_back(vec[1]);
            qVec.push_back(vec[2]);
            qvar = QVariant::fromValue(qVec);
            break;
        }
        case 9:
        {
            zeno::vec4i vec = std::get<zeno::vec4i>(var);
            UI_VECTYPE qVec;
            qVec.push_back(vec[0]);
            qVec.push_back(vec[1]);
            qVec.push_back(vec[2]);
            qVec.push_back(vec[3]);
            qvar = QVariant::fromValue(qVec);
            break;
        }
        case 10: {
            zeno::vec4f vec = std::get<zeno::vec4f>(var);
            UI_VECTYPE qVec;
            qVec.push_back(vec[0]);
            qVec.push_back(vec[1]);
            qVec.push_back(vec[2]);
            qVec.push_back(vec[3]);
            qvar = QVariant::fromValue(qVec);
            break;
        }
    }
    return qvar;
}