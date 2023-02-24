#ifndef __ZENO_EDITOR_DOCKWIDGET_H__
#define __ZENO_EDITOR_DOCKWIDGET_H__

#if 0
#include "zenodockwidget.h"

class ZenoDockTitleWidget : public QWidget
{
	Q_OBJECT
public:
	ZenoDockTitleWidget(QWidget* parent = nullptr);
	~ZenoDockTitleWidget();
	QSize sizeHint() const override;
	void updateByType(DOCK_TYPE type);
	virtual void setupUi();

signals:
	void dockOptionsClicked();
	void dockSwitchClicked(DOCK_TYPE);
    void doubleClicked();
	void actionTriggered(QAction* action);

protected:
	void paintEvent(QPaintEvent* event) override;
	void mouseDoubleClickEvent(QMouseEvent* event) override;
	virtual void initTitleContent(QHBoxLayout* pHLayout);
	QAction* createAction(const QString& text);

private slots:
	void onDockSwitchClicked();
};

class IGraphsModel;

class ZenoEditorDockTitleWidget : public ZenoDockTitleWidget
{
	Q_OBJECT
public:
	ZenoEditorDockTitleWidget(QWidget* parent = nullptr);
	~ZenoEditorDockTitleWidget();
	void initModel();

public slots:
	void setTitle(const QString& title);
	void onModelInited(IGraphsModel* pModel);
	void onModelClear();
	void onPathChanged(const QString& newPath);
	void onDirtyChanged();

protected:
	void initTitleContent(QHBoxLayout* pHLayout) override;
	void paintEvent(QPaintEvent* event) override;

private:
	QMenuBar* initMenu();

	QLabel* m_lblTitle;
};

class ZenoViewDockTitle : public ZenoDockTitleWidget
{
	Q_OBJECT
public:
	ZenoViewDockTitle(QWidget* parent = nullptr);
	~ZenoViewDockTitle();

protected:
	void initTitleContent(QHBoxLayout* pHLayout) override;

private:
	QMenuBar* initMenu();

	QAction* m_pSolidMode;
	QAction* m_pShadingMode;
	QAction* m_pOptixMode;
};

class ZenoPropDockTitleWidget : public ZenoDockTitleWidget
{
	Q_OBJECT
public:
	ZenoPropDockTitleWidget(QWidget* parent = nullptr);
	~ZenoPropDockTitleWidget();

public slots:
	void setTitle(const QString& title);

protected:
	void paintEvent(QPaintEvent* event) override;

private:
	QString m_title;
};

#endif

#endif