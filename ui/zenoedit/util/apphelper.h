#ifndef __ZENOEDIT_HELPER__
#define __ZENOEDIT_HELPER__

#include <zenoui/include/igraphsmodel.h>

class AppHelper
{
public:
    static QString correctSubIOName(IGraphsModel* pModel, const QString& subgName, const QString& newName, bool bInput);
    static QModelIndexList getSubInOutNode(IGraphsModel* pModel, const QModelIndex& subgIdx, const QString& sockName, bool bInput);
    static void reAllocIdents(QMap<QString, NODE_DATA>& nodes, QList<EdgeInfo>& links);
    static QString nthSerialNumName(QString name);
    static QLinearGradient colorString2Grad(const QString& colorStr);
    static QString gradient2colorString(const QLinearGradient& grad);
    static INPUT_SOCKET getInputSocket(const QPersistentModelIndex& index, const QString& inSock, bool& exist);
    static void ensureSRCDSTlastKey(INPUT_SOCKETS& inputs, OUTPUT_SOCKETS& outputs);
};


#endif