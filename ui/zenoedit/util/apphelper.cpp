#include "apphelper.h"
#include <zenoui/model/modeldata.h>
#include <zenoui/model/modelrole.h>
#include "util/log.h"
#include <zenoui/util/uihelper.h>
#include <zenoui/util/cihou.h>


QString AppHelper::correctSubIOName(IGraphsModel* pModel, const QString& subgName, const QString& newName, bool bInput)
{
    ZASSERT_EXIT(pModel, "");

    NODE_DESCS descs = pModel->descriptors();
    if (descs.find(subgName) == descs.end())
        return "";

    const NODE_DESC &desc = descs[subgName];
    QString finalName = newName;
    int i = 1;
    if (bInput)
    {
        while (desc.inputs.find(finalName) != desc.inputs.end())
        {
            finalName = finalName + QString("_%1").arg(i);
            i++;
        }
    }
    else
    {
        while (desc.outputs.find(finalName) != desc.outputs.end())
        {
            finalName = finalName + QString("_%1").arg(i);
            i++;
        }
    }
    return finalName;
}


QModelIndexList AppHelper::getSubInOutNode(IGraphsModel* pModel, const QModelIndex& subgIdx, const QString& sockName, bool bInput)
{
    //get SubInput/SubOutput Node by socket of a subnet node.
    const QModelIndexList& indices = pModel->searchInSubgraph(bInput ? "SubInput" : "SubOutput", subgIdx);
    QModelIndexList result;
    for (const QModelIndex &idx_ : indices)
    {
        const QString& subInputId = idx_.data(ROLE_OBJID).toString();
        if ((sockName == "DST" && !bInput) || (sockName == "SRC" && bInput))
        {
            result.append(idx_);
            continue;
        }
        const PARAMS_INFO &params = idx_.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
        if (params["name"].value == sockName)
        {
            result.append(idx_);
            // there muse be a unique SubOutput for specific name.
            return result;
        }
    }
    return result;
}

void AppHelper::reAllocIdents(QMap<QString, NODE_DATA>& nodes, QList<EdgeInfo>& links)
{
    QMap<QString, QString> old2new;
    QMap<QString, NODE_DATA> newNodes;
    for (QString key : nodes.keys()) {
        const NODE_DATA data = nodes[key];
        const QString &oldId = data[ROLE_OBJID].toString();
        const QString &name = data[ROLE_OBJNAME].toString();
        const QString &newId = UiHelper::generateUuid(name);
        NODE_DATA newData = data;
        newData[ROLE_OBJID] = newId;
        newNodes.insert(newId, newData);
        old2new.insert(oldId, newId);
    }
    //replace all the old-id in newNodes.
    for (QString newId : newNodes.keys()) {
        NODE_DATA &data = newNodes[newId];
        INPUT_SOCKETS inputs = data[ROLE_INPUTS].value<INPUT_SOCKETS>();
        for (INPUT_SOCKET& inputSocket : inputs) {
            inputSocket.info.nodeid = newId;
            inputSocket.linkIndice.clear();
            inputSocket.outNodes.clear();
        }

        OUTPUT_SOCKETS outputs = data[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
        for (OUTPUT_SOCKET& outputSocket : outputs) {
            outputSocket.info.nodeid = newId;
            outputSocket.linkIndice.clear();
            outputSocket.inNodes.clear();
        }

        data[ROLE_INPUTS] = QVariant::fromValue(inputs);
        data[ROLE_OUTPUTS] = QVariant::fromValue(outputs);
    }

    for (EdgeInfo &link : links) {
        ZASSERT_EXIT(old2new.find(link.inputNode) != old2new.end() && old2new.find(link.outputNode) != old2new.end());
        link.inputNode = old2new[link.inputNode];
        link.outputNode = old2new[link.outputNode];
        ZASSERT_EXIT(newNodes.find(link.inputNode) != newNodes.end() &&
                     newNodes.find(link.outputNode) != newNodes.end());
    }

    nodes = newNodes;
}

QString AppHelper::nthSerialNumName(QString name)
{
    QRegExp rx("\\((\\d+)\\)");
    int idx = rx.lastIndexIn(name);
    if (idx == -1) {
        return name + "(1)";
    }
    else {
        name = name.mid(0, idx);
        QStringList lst = rx.capturedTexts();
        ZASSERT_EXIT(lst.size() == 2, "");
        bool bConvert = false;
        int ith = lst[1].toInt(&bConvert);
        ZASSERT_EXIT(bConvert, "");
        return name + "(" + QString::number(ith + 1) + ")";
    }
}

QLinearGradient AppHelper::colorString2Grad(const QString& colorStr)
{
    QLinearGradient grad;
    QStringList L = colorStr.split("\n", QtSkipEmptyParts);
    ZASSERT_EXIT(!L.isEmpty(), grad);

    bool bOk = false;
    int n = L[0].toInt(&bOk);
    ZASSERT_EXIT(bOk && n == L.size() - 1, grad);
    for (int i = 1; i <= n; i++)
    {
        QStringList color_info = L[i].split(" ", QtSkipEmptyParts);
        ZASSERT_EXIT(color_info.size() == 4, grad);

        float f = color_info[0].toFloat(&bOk);
        ZASSERT_EXIT(bOk, grad);
        float r = color_info[1].toFloat(&bOk);
        ZASSERT_EXIT(bOk, grad);
        float g = color_info[2].toFloat(&bOk);
        ZASSERT_EXIT(bOk, grad);
        float b = color_info[3].toFloat(&bOk);
        ZASSERT_EXIT(bOk, grad);

        QColor clr;
        clr.setRgbF(r, g, b);
        grad.setColorAt(f, clr);
    }
    return grad;
}

QString AppHelper::gradient2colorString(const QLinearGradient& grad)
{
    QString colorStr;
    const QGradientStops& stops = grad.stops();
    colorStr += QString::number(stops.size());
    colorStr += "\n";
    for (QGradientStop grad : stops)
    {
        colorStr += QString::number(grad.first);
        colorStr += " ";
        colorStr += QString::number(grad.second.redF());
        colorStr += " ";
        colorStr += QString::number(grad.second.greenF());
        colorStr += " ";
        colorStr += QString::number(grad.second.blueF());
        colorStr += "\n";
    }
    return colorStr;
}
