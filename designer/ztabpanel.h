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
    void setModel(QStandardItemModel *model, QItemSelectionModel *selection);

private:
    ZPagePropPanel* m_pagePanel;
    ZComponentPropPanel* m_componentPanel;
    ZElementPropPanel* m_elementPanel;
};

#endif