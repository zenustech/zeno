#ifndef __CALLBACK_DEF_H__
#define __CALLBACK_DEF_H__

typedef std::function<void(QString, QString)> Callback_EditContentsChange;

class ZenoSocketItem;

typedef std::function<void(ZenoSocketItem*)> Callback_OnSockClicked;

typedef std::function<void(QVariant state)> Callback_EditFinished;

typedef std::function<void(bool bOn)> CALLBACK_SWITCH;

typedef std::function<void(const QModelIndex& idx)> Callback_NodeSelected;

typedef std::function<QVariant()> Callback_GetIndexData;

typedef std::function<QPointF()> Callback_UpdateSockItemPos;
typedef std::function<void()> Callback_OnSockLayoutChanged;

struct CallbackForSocket
{
    Callback_OnSockClicked cbOnSockClicked;
    Callback_OnSockLayoutChanged cbOnSockLayoutChanged;
};

struct CallbackCollection
{
    Callback_EditFinished cbEditFinished;
    CALLBACK_SWITCH cbSwitch;
    Callback_NodeSelected cbNodeSelected;
    Callback_GetIndexData cbGetIndexData;
};

#endif