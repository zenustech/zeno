#include "apphelper.h"
#include <zenoui/model/modeldata.h>
#include <zenoui/model/modelrole.h>
#include "util/log.h"
#include <zenoui/util/uihelper.h>
#include <zenoui/util/cihou.h>


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

INPUT_SOCKET AppHelper::getInputSocket(const QPersistentModelIndex& index, const QString& inSock, bool& exist)
{
    INPUT_SOCKETS inputs = index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    //assuming inSock is valid...
    INPUT_SOCKET _inSocket;
    if (inputs.find(inSock) == inputs.end())
    {
        exist = false;
        return _inSocket;
    }
    _inSocket = inputs[inSock];
    exist = true;
    return _inSocket;
}

void AppHelper::ensureSRCDSTlastKey(INPUT_SOCKETS& inputs, OUTPUT_SOCKETS& outputs)
{
    if (inputs.lastKey() != "SRC")
    {
        //ensure that the "SRC" is the last key in sockets.
        INPUT_SOCKET srcSocket = inputs["SRC"];
        inputs.remove("SRC");
        inputs.insert("SRC", srcSocket);
    }
    if (outputs.lastKey() != "DST")
    {
        //ensure that the "DST" is the last key in sockets.
        OUTPUT_SOCKET dstSocket = outputs["DST"];
        outputs.remove("DST");
        outputs.insert("DST", dstSocket);
    }
}
