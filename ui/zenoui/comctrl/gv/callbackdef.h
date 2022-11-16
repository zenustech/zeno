#ifndef __CALLBACK_DEF_H__
#define __CALLBACK_DEF_H__

typedef std::function<void(QString, QString)> Callback_EditContentsChange;

class ZenoSocketItem;

typedef std::function<void(ZenoSocketItem*)> Callback_OnSockClicked;

typedef std::function<void(QVariant state)> Callback_EditFinished;

typedef std::function<void(bool bOn)> CALLBACK_SWITCH;

#endif