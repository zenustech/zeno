#include "ZenoDictListLInksPanel.h"
#include <set>

IconDelegate::IconDelegate(bool bfirst, QObject* parent) : m_bFirstColumn(bfirst), QStyledItemDelegate(parent)
{
}

void IconDelegate::paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    QIcon icon = QApplication::style()->standardIcon(m_bFirstColumn ? QStyle::SP_TitleBarShadeButton : QStyle::SP_MessageBoxCritical);
    if (!icon.isNull()) {
        QRect iconRect = option.rect;
        iconRect.adjust(6, 6, -6, -6);
        icon.paint(painter, iconRect, Qt::AlignCenter);
    }
    else {
        QStyledItemDelegate::paint(painter, option, index);
    }
}

DragDropModel::DragDropModel(int row, int column, int allowDragColumn, QObject* parent) : QAbstractTableModel(parent), m_allowDragColumn(allowDragColumn)
{
    dataMatrix = QVector<QVector<QString>>(row, QVector<QString>(column, ""));
    dataMatrix =
    {
        {"", "R1-C1", "R1-C2", ""},
        {"", "R2-C1", "R2-C2", ""},
        {"", "R3-C1", "R3-C2", ""},
        {"", "R4-C1", "R4-C2", ""}
    };
    horizontalHeaderLabels << "" << "key" << "out node" << "";
}

int DragDropModel::rowCount(const QModelIndex& parent) const
{
    return dataMatrix.size();
}

int DragDropModel::columnCount(const QModelIndex& parent) const
{
    return dataMatrix.isEmpty() ? 0 : dataMatrix.first().size();
}

QVariant DragDropModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid() || role != Qt::DisplayRole) {
        return QVariant();
    }
    return dataMatrix[index.row()][index.column()];
}

bool DragDropModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    if (index.isValid() && role == Qt::DisplayRole) {
        dataMatrix[index.row()][index.column()] = value.toString();
        emit dataChanged(index, index);
        return true;
    }
    return false;
}

bool DragDropModel::removeRows(int row, int count, const QModelIndex& parent)
{
    if (count <= 0 || row < 0 || row + count > dataMatrix.size())
        return false;
    beginRemoveRows(parent, row, row + count - 1);
    for (int i = 0; i < count; ++i) {
        dataMatrix.removeAt(row + i);
    }
    endRemoveRows();
    return true;
}

Qt::ItemFlags DragDropModel::flags(const QModelIndex& index) const
{
    if (!index.isValid()) return Qt::NoItemFlags;
    return QAbstractTableModel::flags(index) | Qt::ItemIsDragEnabled | Qt::ItemIsDropEnabled;
}

QVariant DragDropModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (role != Qt::DisplayRole) {
        return QVariant();
    }
    if (orientation == Qt::Horizontal && section < horizontalHeaderLabels.size()) {
        return horizontalHeaderLabels[section];
    }
    return QVariant();
}

Qt::DropActions DragDropModel::supportedDropActions() const
{
    return Qt::MoveAction | QAbstractTableModel::supportedDropActions();
}

QMimeData* DragDropModel::mimeData(const QModelIndexList& indexes) const
{
    QMimeData* mimeData = QAbstractTableModel::mimeData(indexes);
    QByteArray data;
    QDataStream stream(&data, QIODevice::WriteOnly);
    foreach(const QModelIndex & index, indexes) {
        if (index.isValid()) {
            if (index.column() != m_allowDragColumn)
                return nullptr;
            stream << index.row();
        }
    }
    mimeData->setData("rows", data);
    return mimeData;
}

bool DragDropModel::dropMimeData(const QMimeData* data, Qt::DropAction action, int row, int column, const QModelIndex& parent)
{
    if (!data || action != Qt::MoveAction)
        return false;
    if (parent.column() != m_allowDragColumn)
        return false;

    QDataStream stream(data->data("rows"));
    std::set<int> selectedRows;
    int r = 0, targetRow = parent.row();
    while (!stream.atEnd()) {
        stream >> r;
        selectedRows.insert(r);
    }
    m_reorderedTexts.clear();
    QList<QString> selectedTexts;
    for (auto& i : selectedRows) {
        selectedTexts.append(index(i, m_allowDragColumn).data(Qt::DisplayRole).toString());
    }
    for (int i = 0; i < rowCount(); i++) {
        if (!selectedRows.count(i)) {
            m_reorderedTexts.append(index(i, m_allowDragColumn).data(Qt::DisplayRole).toString());
        }
    }
    if (parent.row() < m_reorderedTexts.size()) {
        for (int i = selectedTexts.size() - 1; i >= 0; i--) {
            m_reorderedTexts.insert(parent.row(), selectedTexts[i]);
        }
    }
    else {
        m_reorderedTexts.append(selectedTexts);
    }
    for (int i = 0; i < dataMatrix.size(); i++) {
        dataMatrix[i][m_allowDragColumn] = m_reorderedTexts[i];
    }
    return true;
}

ZenoDictListLinksTable::ZenoDictListLinksTable(int row, int column, int allowDragColumn, QWidget* parent) : QTableView(parent), m_model(nullptr), m_allowDragColumn(allowDragColumn)
{
    //drag
    verticalHeader()->setVisible(false);
    setDragEnabled(true);
    viewport()->setAcceptDrops(true);
    setDragDropMode(QAbstractItemView::InternalMove);
    //model
    m_model = new DragDropModel(row, column, m_allowDragColumn, this);
    setModel(m_model);

    //delegateIcon
    IconDelegate* firstColumnDelegate = new IconDelegate(true, this);
    setItemDelegateForColumn(0, firstColumnDelegate);
    IconDelegate* lastColumnDelegate = new IconDelegate(false, this);
    setItemDelegateForColumn(m_model->columnCount() - 1, lastColumnDelegate);

    horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    horizontalHeader()->setSectionResizeMode(m_model->columnCount() - 1, QHeaderView::ResizeToContents);

    connect(this, &QTableView::clicked, this, &ZenoDictListLinksTable::slt_clicked);
}

void ZenoDictListLinksTable::dragEnterEvent(QDragEnterEvent* event)
{
    if (event->source() != this) {  // 允许拖动进入视图
        event->ignore();
        return;
    }
    QModelIndex index = indexAt(event->pos());
    if (index.isValid() && index.column() == m_allowDragColumn) {   // 允许拖放到allowDragColumn
        event->acceptProposedAction();
    }
    else {
        event->setDropAction(Qt::IgnoreAction);
        event->ignore();
    }
}

void ZenoDictListLinksTable::dragMoveEvent(QDragMoveEvent* event)
{
    QModelIndex index = indexAt(event->pos());
    if (index.isValid() && index.column() == m_allowDragColumn) {   // 允许拖放到allowDragColumn
        event->acceptProposedAction();
    }
    else {
        event->setDropAction(Qt::IgnoreAction);
        event->ignore();
    }
}

void ZenoDictListLinksTable::dropEvent(QDropEvent* event)
{
    QModelIndex index = indexAt(event->pos());
    if (index.isValid() && index.column() == m_allowDragColumn) {   // 允许拖放到allowDragColumn
        QTableView::dropEvent(event);
        selectionModel()->clearSelection();
    }
    else {
        event->ignore();
    }
}

void ZenoDictListLinksTable::slt_clicked(const QModelIndex& index)
{
    int row = index.row();
    int column = index.column();
    if (column == 0) {
        if (row != 0) {
            QString uptext = m_model->data(m_model->index(row - 1, m_allowDragColumn), Qt::DisplayRole).toString();
            m_model->setData(m_model->index(row - 1, m_allowDragColumn), m_model->data(m_model->index(row, m_allowDragColumn), Qt::DisplayRole).toString(), Qt::DisplayRole);
            m_model->setData(m_model->index(row, m_allowDragColumn), uptext, Qt::DisplayRole);
        }
    }
    else if (column == m_model->columnCount() - 1) {
        m_model->removeRows(row, 1);
    }
}
