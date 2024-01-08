#ifndef __ZDICT_PANEL_H__
#define __ZDICT_PANEL_H__

#include "zlayoutbackground.h"
#include "callbackdef.h"

class ZDictSocketLayout;
class ZenoParamPushButton;
class IGraphsModel;

class ZDictPanel : public ZLayoutBackground
{
    Q_OBJECT
public:
    ZDictPanel(ZDictSocketLayout* pLayout, const QPersistentModelIndex& viewSockIdx, const CallbackForSocket& cbSock, IGraphsModel* pModel);
    ~ZDictPanel();
    ZenoSocketItem* socketItemByIdx(const QModelIndex& sockIdx) const;
    IGraphsModel* graphsModel() const;
    QModelIndex dictlistSocket() const;

private slots:
    void onKeysAboutToBeRemoved(const QModelIndex& parent, int first, int last);
    void onKeysMoved(const QModelIndex& parent, int start, int end, const QModelIndex& destination, int row);
    void onKeysInserted(const QModelIndex& parent, int first, int last);
    void onKeysModelDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles);
    void onAddRemoveLink(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles);
    void onEditBtnClicked();

private:
    void setEnable(bool bEnable);

    const QPersistentModelIndex m_viewSockIdx;
    CallbackForSocket m_cbSock;
    IGraphsModel* m_model;
    ZDictSocketLayout* m_pDictLayout;
    ZenoParamPushButton* m_pEditBtn;
    bool m_bDict;
};

#endif