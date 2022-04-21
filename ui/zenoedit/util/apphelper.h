#ifndef __ZENOEDIT_HELPER__
#define __ZENOEDIT_HELPER__

#include <zenoui/include/igraphsmodel.h>

class AppHelper
{
public:
    static void correctSubIOName(IGraphsModel* pModel, QModelIndex subgIdx, const QString& descName, PARAMS_INFO& params);
};


#endif