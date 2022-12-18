#ifndef __ZDICT_PANEL_H__
#define __ZDICT_PANEL_H__

#include "zlayoutbackground.h"

class ZDictPanel : public ZLayoutBackground
{
    Q_OBJECT
public:
    ZDictPanel(const QPersistentModelIndex& viewSockIdx);

private:
    const QPersistentModelIndex m_viewSockIdx;
};

#endif