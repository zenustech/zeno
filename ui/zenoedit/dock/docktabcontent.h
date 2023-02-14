#ifndef __DOCKTAB_CONTENT_H__
#define __DOCKTAB_CONTENT_H__

#include <QtWidgets>
#include <unordered_set>
#include <zenoui/comctrl/ztoolbutton.h>

class ZIconToolButton;
class ZenoGraphsEditor;
class ZTextLabel;

class ZToolBarButton : public ZToolButton
{
    Q_OBJECT
public:
    ZToolBarButton(bool bCheckable, const QString& icon, const QString& iconOn);
};

class DockToolbarWidget : public QWidget
{
    Q_OBJECT
public:
    explicit DockToolbarWidget(QWidget* parent = nullptr);
    QWidget* widget() const;
    virtual void initUI();

protected:
    virtual void initToolbar(QHBoxLayout* pToolLayout) = 0;
    virtual QWidget *initWidget() = 0;
    virtual void initConnections() = 0;

    QWidget* m_pWidget;
};

class DockContent_Parameter : public DockToolbarWidget
{
    Q_OBJECT
public:
    explicit DockContent_Parameter(QWidget* parent = nullptr);
    void onNodesSelected(const QModelIndex& subgIdx, const QModelIndexList& nodes, bool select);
    void onPrimitiveSelected(const std::unordered_set<std::string>& primids);

protected:
    void initToolbar(QHBoxLayout* pToolLayout) override;
    QWidget* initWidget() override;
    void initConnections() override;

private:
    QLabel* m_plblName;
    QLineEdit* m_pLineEdit;
    ZToolBarButton* m_pSettingBtn;
};

class DockContent_Editor : public DockToolbarWidget
{
    Q_OBJECT
public:
    explicit DockContent_Editor(QWidget* parent = nullptr);
    void onCommandDispatched(QAction* pAction, bool bTriggered);
    ZenoGraphsEditor* getEditor() const;

protected:
    void initToolbar(QHBoxLayout* pToolLayout) override;
    QWidget* initWidget() override;
    void initConnections() override;

private:
    ZenoGraphsEditor* m_pEditor;
    ZTextLabel* lblFileName;
    ZToolBarButton *pListView;
    ZToolBarButton *pTreeView;
    ZToolBarButton *pSubnetMgr;
    ZToolBarButton *pFold;
    ZToolBarButton *pUnfold;
    ZToolBarButton *pSnapGrid;
    ZToolBarButton *pBlackboard;
    ZToolBarButton *pFullPanel;
    ZToolBarButton *pSearchBtn;
    ZToolBarButton *pSettings;
    QComboBox* cbZoom;
};

class DockContent_View : public QWidget
{
    Q_OBJECT
public:
    explicit DockContent_View(QWidget* parent = nullptr);
};

class DockContent_Log : public DockToolbarWidget
{
    Q_OBJECT
public:
    explicit DockContent_Log(QWidget* parent = nullptr);

protected:
    void initToolbar(QHBoxLayout* pToolLayout) override;
    QWidget* initWidget() override;
    void initConnections() override;

private:
    QStackedWidget* m_stack;
    ZToolBarButton* m_pBtnFilterLog;
    ZToolBarButton* m_pBtnPlainLog;
};



#endif