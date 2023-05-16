#include "apilevelscope.h"
#include "igraphsmodel.h"


ApiLevelScope::ApiLevelScope(IGraphsModel* pModel)
    : m_model(pModel)
{
    m_model->beginApiLevel();
}

ApiLevelScope::~ApiLevelScope()
{
    m_model->endApiLevel();
}