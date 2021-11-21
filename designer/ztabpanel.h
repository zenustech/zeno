#ifndef __ZTABPANEL_H__
#define __ZTABPANEL_H__

class ZTabPanel : public QTabWidget
{
    Q_OBJECT
public:
    ZTabPanel(QWidget* parent = nullptr);
    ~ZTabPanel();
};

#endif