#ifndef __STYLE_TABWIDGET_H__
#define __STYLE_TABWIDGET_H__

#include "renderparam.h"
#include "common.h"

class LayerWidget;

class StyleTabWidget : public QTabWidget
{
    Q_OBJECT
public:
    StyleTabWidget(QWidget* parent = nullptr);
    QStandardItemModel* getCurrentModel();
    QItemSelectionModel* getSelectionModel();

signals:
    void tabClosed(int);
    void tabActivate(NodeParam);
	void imageElemOperated(ImageElement, NODE_ID);
	void textElemOperated(TextElement, NODE_ID);
	void compElementOperated(NODE_OPERATE, NODE_ID);

    void tabviewActivated(QStandardItemModel*);

public slots:
    void onNewTab();
    void onTabClosed(int);

private:
    void initTabs();

};

#endif