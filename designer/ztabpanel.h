#ifndef __ZTABPANEL_H__
#define __ZTABPANEL_H__

#include "zpropertiespanel.h"

class ZTabPanel : public QTabWidget
{
    Q_OBJECT
public:
    ZTabPanel(QWidget* parent = nullptr);
    ~ZTabPanel();

public slots:
    void resetModel();

private:
    ZPagePropPanel* m_pagePanel;
    ZComponentPropPanel* m_componentPanel;
    ZElementPropPanel* m_elementPanel;
};

#endif