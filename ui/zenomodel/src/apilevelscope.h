#ifndef __API_LEVEL_SCOPE_H__
#define __API_LEVEL_SCOPE_H__

class IGraphsModel;

class ApiLevelScope
{
public:
    ApiLevelScope(IGraphsModel* pModel);
    ~ApiLevelScope();

private:
    IGraphsModel* m_model;
};


#endif