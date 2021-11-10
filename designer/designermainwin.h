#ifndef __DesignerMainWindow_H__
#define __DesignerMainWindow_H__

#include "framework.h"

class LayerWidget;
class ViewMdiArea;
class PropertyPane;
class StyleTabWidget;

class DesignerMainWin : public QMainWindow
{
    Q_OBJECT
public:
    DesignerMainWin();

private:
    void initMenu();
    void initWidgets();
    void initMdiWindows();

    QMenuBar* m_pMenubar;
    LayerWidget* m_pLayerWidget;
    ViewMdiArea* m_pMdiArea;
    PropertyPane* m_properties;
    StyleTabWidget* m_tabWidget;
};

#endif