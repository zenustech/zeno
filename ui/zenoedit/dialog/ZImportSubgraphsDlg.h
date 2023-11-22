#ifndef __ZIMPORT_SUBGRAPHS_DLG_H__
#define __ZIMPORT_SUBGRAPHS_DLG_H__

#include <QtWidgets>
#include "ui_ZImportSubgraphsDlg.h"

class CheckBoxHeaderView : public QHeaderView
{
    Q_OBJECT

public:
    CheckBoxHeaderView(Qt::Orientation orientation, QWidget* parent = nullptr);
    void setCheckState(QVector<int> columns, bool state);

signals:
    void signalCheckStateChanged(int col, bool state);

protected:
    void paintSection(QPainter* painter, const QRect& rect, int logicalIndex) const;
    void mousePressEvent(QMouseEvent* event);

private:
    QMap<int, bool>    m_checkedMap;      //¹´Ñ¡¿ò×´Ì¬
};

class ZSubgraphsListDlg : public QDialog
{
    Q_OBJECT

public:
    ZSubgraphsListDlg(const QStringList& lst, QWidget* parent = nullptr);
    ~ZSubgraphsListDlg();
signals:
    void selectedSignal(const QStringList& lst, bool isRename);
private:
    void initUI(const QStringList& lst);
    void updateCheckState(int col, bool state);
private slots:
    void onOkBtnClicked();
private:
    QTableWidget* m_pTableWidget;
};

class ZImportSubgraphsDlg : public QDialog
{
    Q_OBJECT

public:
    ZImportSubgraphsDlg(const QStringList &lst, QWidget *parent = nullptr);
    ~ZImportSubgraphsDlg();
signals:
    void selectedSignal(const QStringList& lst, bool isRename);
private slots:
    void onSelectBtnClicked();
private:
    Ui::ZImportSubgraphsDlg* m_ui;
    const QStringList& m_subgraphs;
};

#endif
