#ifndef __CALLBACK_DEF_H__
#define __CALLBACK_DEF_H__

typedef std::function<void(QString, QString)> Callback_EditContentsChange;
typedef std::function<void(QString)> Callback_OnClicked;

class ZenoSocketItem;

typedef std::function<void(ZenoSocketItem*, Qt::MouseButton)> Callback_OnSockClicked;

typedef std::function<void(ZenoSocketItem*)> Callback_OnSockLabelEdited;

typedef std::function<void(QVariant state)> Callback_EditFinished;
typedef std::function<void(QVariant state)> Callback_EditFinishedWithSlider;

typedef std::function<void(bool bOn)> CALLBACK_SWITCH;

typedef std::function<void(const QModelIndex& idx)> Callback_NodeSelected;
typedef std::function<void(QAction* pAction, const QModelIndex& idx)> Callback_ActionTriggered;

typedef std::function<QVariant()> Callback_GetIndexData;

typedef std::function<QPointF()> Callback_UpdateSockItemPos;
typedef std::function<void()> Callback_OnSockLayoutChanged;

typedef std::function<void(int nframe)> Callback_UpdateFrame;

struct CallbackForSocket
{
    Callback_OnSockClicked cbOnSockClicked;
    Callback_OnSockLabelEdited cbOnSockNetlabelEdited;
    Callback_OnClicked cbOnSockNetlabelClicked;
    Callback_OnSockLayoutChanged cbOnSockLayoutChanged;
    Callback_ActionTriggered cbActionTriggered;
};

struct CallbackCollection
{
    Callback_EditFinished cbEditFinished;
    Callback_EditFinishedWithSlider cbEditFinishedWithSlider;
    CALLBACK_SWITCH cbSwitch;
    Callback_NodeSelected cbNodeSelected;
    Callback_GetIndexData cbGetIndexData;
};

#endif