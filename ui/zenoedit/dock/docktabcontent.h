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
class ZToolMenuButton;

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


class ZTextIconButton : public QWidget
{
    Q_OBJECT
public:
    ZTextIconButton(const QString &text, QWidget *parent = nullptr);
    ~ZTextIconButton();
    void setShortcut(QKeySequence text);
signals:
    void clicked();

private:
    QPushButton* m_pButton;
    QLabel* m_pLablel;
    QShortcut* m_shortcut;
};

class DockToolbarWidget : public QWidget
{
    Q_OBJECT
public:
    explicit DockToolbarWidget(QWidget* parent = nullptr);
    QWidget* widget() const;
    virtual void initUI();
    virtual void onTabAboutToClose();

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
    void runFinished();

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
    ZToolBarButton *pLinkLineShape;
    QCheckBox*pAlways;
    ZToolBarButton *pSearchBtn;
    ZToolBarButton *pSettings;

    ZToolMenuButton *m_btnRun;
    ZTextIconButton* m_btnKill;

    QComboBox* cbZoom;
    ZComboBox* cbSubgType;
};

class DockContent_View : public DockToolbarWidget
{
    Q_OBJECT
public:
    explicit DockContent_View(bool bGLView, QWidget* parent = nullptr);
    void onCommandDispatched(QAction* pAction, bool bTriggered);
    DisplayWidget* getDisplayWid() const;
    bool isGLView() const;
    QSize viewportSize() const;
    void onTabAboutToClose() override;

    int curResComboBoxIndex();
    void setResComboBoxIndex(int index);
    std::tuple<int, int, bool> getOriginWindowSizeInfo();
    void setOptixBackgroundState(bool checked);

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
    QPushButton *m_camera_setting = nullptr;
    QCheckBox *m_background;
    QCheckBox *m_uv_mode = nullptr;

    QComboBox* m_cbRes;
    QAction* m_pFocus;
    QAction *m_pOrigin;
    QAction *m_front;
    QAction *m_back;
    QAction *m_right;
    QAction *m_left;
    QAction *m_top; 
    QAction *m_bottom;
    
    QMenu* m_menuView;
    QMenu* m_menuViewport;

    const bool m_bGLView;
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
    ZToolBarButton* m_pDeleteLog;
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