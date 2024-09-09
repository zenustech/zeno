#ifndef __ZENO_DICTLIST_PANEL_H__
#define __ZENO_DICTLIST_PANEL_H__

#include <QtWidgets>
#include "uicommon.h"

class IconDelegate : public QStyledItemDelegate {
    Q_OBJECT

public:
    explicit IconDelegate(bool bfirst, QObject* parent = nullptr);

    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
private:
    bool m_bFirstColumn;
};

class DragDropModel : public QAbstractTableModel {
    Q_OBJECT

public:
    explicit DragDropModel(const QModelIndex& inputObjsIdx, int allowDragColumn, QObject* parent = nullptr);
    //base
    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    int columnCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
    bool removeRows(int row, int count, const QModelIndex& parent = QModelIndex()) override;
    Qt::ItemFlags flags(const QModelIndex& index) const override;
    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
    //drag
    Qt::DropActions supportedDropActions() const override;
    QMimeData* mimeData(const QModelIndexList& indexes) const override;
    bool dropMimeData(const QMimeData* data, Qt::DropAction action, int row, int column, const QModelIndex& parent) override;
    //
    void insertLink(const zeno::EdgeInfo& edge);
    QList<QPair<QString, QModelIndex>> linksNeedUpdate();
    QList<QModelIndex> linksNeedRemove();
    QString getInKeyFromOutnodeName(const QString& nodeNameParam);

signals:
    void keyUpdated(QList<QPair<QString, QModelIndex>>);
private:
    int m_allowDragColumn;
    QList<QString> m_reorderedTexts;
    QHash<QString, QString> m_outNodesInkeyMap;

    QStringList horizontalHeaderLabels;
    QList<QList<QString>> dataMatrix;

    QPersistentModelIndex m_objsParamIdx;
};

class ZenoDictListLinksTable : public QTableView {
    Q_OBJECT

public:
    ZenoDictListLinksTable(int allowDragColumn, QWidget* parent = nullptr);
    void init();

    void addLink(const zeno::EdgeInfo& edge);
    void removeLink(const zeno::EdgeInfo& edge);

protected:
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dragMoveEvent(QDragMoveEvent* event) override;
    void dropEvent(QDropEvent* event) override;

signals:
    void linksRemoved(QList<QModelIndex>);
    void linksUpdated(QList<QPair<QString, QModelIndex>>);

private slots:
    void slt_clicked(const QModelIndex& index);
private:
    int m_allowDragColumn;
};

#endif