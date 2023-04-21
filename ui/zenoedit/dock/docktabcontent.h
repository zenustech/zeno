#ifndef __DOCKTAB_CONTENT_H__
#define __DOCKTAB_CONTENT_H__

#include <QtWidgets>
#include <unordered_set>
#include <zenoui/comctrl/ztoolbutton.h>

class ZIconToolButton;
class ZenoGraphsEditor;
class ZenoImagePanel;
class ZTextLabel;
class DisplayWidget;
class ZComboBox;
class ZLineEdit;

class ZToolBarButton : public ZToolButton
{
    Q_OBJECT
public:
    ZToolBarButton(bool bCheckable, const QString& icon, const QString& iconOn);
};

#if 0
class ZToolRecordingButton : public ZToolButton {
    Q_OBJECT
public:
    ZToolRecordingButton(const QString &icon, const QString &iconHover, const QString &iconOn,
                         const QString &iconOnHover, const QString &iconPressed);
protected:
    void paintEvent(QPaintEvent* event) override;
private:
    QIcon m_iconOnPressed;
};
#endif

class ZToolMenuButton : public ZToolButton {
    Q_OBJECT
public:
    ZToolMenuButton();

protected:
    virtual void mouseReleaseEvent(QMouseEvent* e) override;
    virtual QSize sizeHint() const override;
private:
    QMenu *menu;
    QAction *run;
    QAction *runLightCameraMaterial;
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
    static const int sToolbarHeight;
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
    ZToolBarButton* m_pSettingBtn;
    ZLineEdit *m_pNameLineEdit;
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
    ZToolBarButton *pListView;
    ZToolBarButton *pTreeView;
    ZToolBarButton *pSubnetMgr;
    ZToolBarButton *pFold;
    ZToolBarButton *pUnfold;
    ZToolBarButton *pSnapGrid;
    ZToolBarButton *pShowGrid;
    ZToolBarButton *pCustomParam;
    ZToolBarButton *pGroup;
    ZToolBarButton *pSearchBtn;
    ZToolBarButton *pSettings;

    ZToolMenuButton *m_btnRun;
    ZToolButton* m_btnKill;
    ZComboBox *m_btnAlways;

    QComboBox* cbZoom;
};

class DockContent_View : public DockToolbarWidget
{
    Q_OBJECT
public:
    explicit DockContent_View(QWidget* parent = nullptr);
    void onCommandDispatched(QAction* pAction, bool bTriggered);
    DisplayWidget* getDisplayWid() const;
    QSize viewportSize() const;

protected:
    void initToolbar(QHBoxLayout* pToolLayout) override;
    QWidget *initWidget() override;
    void initConnections() override;

private:
    DisplayWidget* m_pDisplay;
    ZToolBarButton* m_smooth_shading;
    ZToolBarButton* m_normal_check;
    ZToolBarButton* m_wire_frame;
    ZToolBarButton* m_show_grid;
    ZToolBarButton* m_background_clr;
    ZToolBarButton *m_recordVideo;
    ZToolBarButton* m_screenshoot;
    ZToolBarButton* m_moveBtn;
    ZToolBarButton* m_scaleBtn;
    ZToolBarButton* m_rotateBtn;
    ZToolBarButton* m_resizeViewport;

    QComboBox* m_cbRenderWay;
    QAction* m_pFocus;
    QAction *m_pOrigin;
    QAction *m_front;
    QAction *m_back;
    QAction *m_right;
    QAction *m_left;
    QAction *m_top;
    QAction *m_bottom;

    QAction *m_move;
    QAction *m_rotate;
    QAction *m_scale;
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

class DockContent_Image : public DockToolbarWidget {
    Q_OBJECT
public:
    explicit DockContent_Image(QWidget *parent = nullptr);
    ZenoImagePanel *getImagePanel();

protected:
    QWidget *initWidget() override;
    void initToolbar(QHBoxLayout *pToolLayout) override{};
    void initConnections() override{};

private:
    ZenoImagePanel *m_ImagePanel;
};

#endif