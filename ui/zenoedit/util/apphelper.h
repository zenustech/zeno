#ifndef __ZENOEDIT_HELPER__
#define __ZENOEDIT_HELPER__

#include <zenoui/include/igraphsmodel.h>

class AppHelper
{
public:
    static QString correctSubIOName(IGraphsModel* pModel, const QString& subgName, const QString& newName, bool bInput);
};


#endif