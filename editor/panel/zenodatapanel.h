#ifndef __ZENO_DATAPANEL_H__
#define __ZENO_DATAPANEL_H__

#include <QtWidgets>

class NodeDataModel;

class ZenoDataTable : public QTableView
{
	Q_OBJECT
public:
    ZenoDataTable(QWidget* parent = nullptr);
    ~ZenoDataTable();

private:
    void init();

    NodeDataModel* m_model;
};

class ZenoDataPanel : public QWidget
{
    Q_OBJECT
public:
    ZenoDataPanel(QWidget* parent = nullptr);
    ~ZenoDataPanel();

private:
    void init();
};

#endif