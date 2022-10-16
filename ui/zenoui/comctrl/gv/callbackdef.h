#ifndef __CALLBACK_DEF_H__
#define __CALLBACK_DEF_H__

typedef std::function<void(QString, QString)> Callback_EditContentsChange;

class ZenoSocketItem;

typedef std::function<void(ZenoSocketItem*)> Callback_OnSockClicked;


#endif