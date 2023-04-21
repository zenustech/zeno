#ifndef __ZENOEDIT_HELPER__
#define __ZENOEDIT_HELPER__

#include <zenomodel/include/igraphsmodel.h>

class AppHelper
{
public:
    static QModelIndexList getSubInOutNode(IGraphsModel* pModel, const QModelIndex& subgIdx, const QString& sockName, bool bInput);
    static QLinearGradient colorString2Grad(const QString& colorStr);
    static INPUT_SOCKET getInputSocket(const QPersistentModelIndex& index, const QString& inSock, bool& exist);
    static void ensureSRCDSTlastKey(INPUT_SOCKETS& inputs, OUTPUT_SOCKETS& outputs);
    static QString nativeWindowTitle(const QString& currentFilePath);
};


#endif