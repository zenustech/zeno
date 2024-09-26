#include "ZenoDictListLInksPanel.h"
#include <zeno/core/data.h>
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

DragDropModel::DragDropModel(const QModelIndex& inputObjsIdx, int allowDragColumn, QObject* parent) : m_objsParamIdx(inputObjsIdx), m_allowDragColumn(allowDragColumn), QAbstractTableModel(parent)
{
    QList<QPersistentModelIndex> linksIdx = m_objsParamIdx.data(ROLE_LINKS).value<QList<QPersistentModelIndex>>();
    for (auto& idx : linksIdx) {
        insertLink(idx.data(ROLE_LINK_INFO).value<zeno::EdgeInfo>());
    }
    horizontalHeaderLabels << "" << "key" << "out node" << "";
}

int DragDropModel::rowCount(const QModelIndex& parent) const
{
    return dataMatrix.size();
}

int DragDropModel::columnCount(const QModelIndex& parent) const
{
    return dataMatrix.isEmpty() ? 4 : dataMatrix.first().size();
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
    if (!index.isValid()) {
        return false;
    }
    if (role == Qt::DisplayRole) {
        m_outNodesInkeyMap[value.toString()] = dataMatrix[index.row()][1];
        dataMatrix[index.row()][index.column()] = value.toString();
        emit dataChanged(index, index);
        return true;
    } else if (role == Qt::EditRole) {
        QString newkey = value.toString();
        const auto& existkey = [this, &newkey]() -> bool {
            for (int i = 0; i < dataMatrix.size(); i++) {
                if (dataMatrix[i][1] == newkey)
                    return true;
            }
            return false;
        };
        while (existkey()) {
            newkey += "_duplicate";
        }
        m_outNodesInkeyMap[dataMatrix[index.row()][m_allowDragColumn]] = newkey;
        dataMatrix[index.row()][index.column()] = newkey;
        emit keyUpdated(linksNeedUpdate());
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
        m_outNodesInkeyMap.remove(dataMatrix[row + i][m_allowDragColumn]);
        dataMatrix.removeAt(row + i);
    }
    endRemoveRows();
    return true;
}

Qt::ItemFlags DragDropModel::flags(const QModelIndex& index) const
{
    if (!index.isValid()) return Qt::NoItemFlags;
    if (index.column() == 1) {
        return QAbstractTableModel::flags(index) | Qt::ItemIsDragEnabled | Qt::ItemIsDropEnabled | Qt::ItemIsEditable;
    }
    else {
        return QAbstractTableModel::flags(index) | Qt::ItemIsDragEnabled | Qt::ItemIsDropEnabled;
    }
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
        m_outNodesInkeyMap[m_reorderedTexts[i]] = dataMatrix[i][1];
        dataMatrix[i][m_allowDragColumn] = m_reorderedTexts[i];
    }
    return true;
}

void DragDropModel::insertLink(const zeno::EdgeInfo& edge)
{
    if (m_outNodesInkeyMap.contains(QString::fromStdString(edge.outNode + ":" + edge.outParam))) {
        return;
    }
    const auto& linkinfo = QList<QString>({ "", QString::fromStdString(edge.inKey), QString::fromStdString(edge.outNode + ":" + edge.outParam), "" });
    if (dataMatrix.empty()) {
        beginInsertRows(QModelIndex(), 0, 0);
        dataMatrix.push_back(linkinfo);
    }
    else {
        int last = dataMatrix.size() - 1;
        if (dataMatrix[last][1] > QString::fromStdString(edge.inKey)) {
            while (last >= 0 && dataMatrix[last][1] > QString::fromStdString(edge.inKey)) {
                last -= 1;
            }
            beginInsertRows(QModelIndex(), last + 1, last + 1);
            dataMatrix.insert(last + 1, linkinfo);
        }
        else {
            beginInsertRows(QModelIndex(), dataMatrix.size(), dataMatrix.size());
            dataMatrix.push_back(linkinfo);
        }
    }
    endInsertRows();
    m_outNodesInkeyMap.insert(QString::fromStdString(edge.outNode + ":" + edge.outParam), QString::fromStdString(edge.inKey));
}

QList<QPair<QString, QModelIndex>> DragDropModel::linksNeedUpdate()
{
    QList<QPair<QString, QModelIndex>> updateLinks;
    QList<QPersistentModelIndex> currLinkIdxs = m_objsParamIdx.data(ROLE_LINKS).value<QList<QPersistentModelIndex>>();
    for (auto& link : currLinkIdxs) {
        zeno::EdgeInfo edge = link.data(ROLE_LINK_INFO).value<zeno::EdgeInfo>();
        auto inkey = getInKeyFromOutnodeName(QString::fromStdString(edge.outNode + ":" + edge.outParam));
        if (inkey != "" && inkey.toStdString() != edge.inKey) {
            updateLinks.append({inkey, link});
        }
    }
    return updateLinks;
}

QList<QModelIndex> DragDropModel::linksNeedRemove()
{
    QList<QModelIndex> removeLinks;
    QList<QPersistentModelIndex> currLinkIdxs = m_objsParamIdx.data(ROLE_LINKS).value<QList<QPersistentModelIndex>>();
    for (auto& link : currLinkIdxs) {
        zeno::EdgeInfo edge = link.data(ROLE_LINK_INFO).value<zeno::EdgeInfo>();
        auto inkey = getInKeyFromOutnodeName(QString::fromStdString(edge.outNode + ":" + edge.outParam));
        if (inkey == "") {
            removeLinks.append(link);
        }
    }
    return removeLinks;
}

QString DragDropModel::getInKeyFromOutnodeName(const QString& nodeNameParam)
{
    return m_outNodesInkeyMap.contains(nodeNameParam) ? m_outNodesInkeyMap.value(nodeNameParam) : "";
}

ZenoDictListLinksTable::ZenoDictListLinksTable(int allowDragColumn, QWidget* parent) : QTableView(parent), m_allowDragColumn(allowDragColumn)
{
    //drag
    verticalHeader()->setVisible(false);
    setDragEnabled(true);
    viewport()->setAcceptDrops(true);
    setDragDropMode(QAbstractItemView::InternalMove);

    connect(this, &QTableView::clicked, this, &ZenoDictListLinksTable::slt_clicked);
}

void ZenoDictListLinksTable::init()
{
    QAbstractItemModel* m_model = model();
    //delegateIcon
    IconDelegate* firstColumnDelegate = new IconDelegate(true, this);
    setItemDelegateForColumn(0, firstColumnDelegate);
    IconDelegate* lastColumnDelegate = new IconDelegate(false, this);
    setItemDelegateForColumn(m_model->columnCount() - 1, lastColumnDelegate);

    horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    if (m_model->columnCount() > 0) {
        horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
        horizontalHeader()->setSectionResizeMode(m_model->columnCount() - 1, QHeaderView::ResizeToContents);
    }

    if (DragDropModel* model = qobject_cast<DragDropModel*>(this->model())) {
        connect(model, &DragDropModel::keyUpdated, this, &ZenoDictListLinksTable::linksUpdated);
    }
}

void ZenoDictListLinksTable::addLink(const zeno::EdgeInfo& edge)
{
    if (DragDropModel* model = qobject_cast<DragDropModel*>(this->model())){
        model->insertLink(edge);
    }
}

void ZenoDictListLinksTable::removeLink(const zeno::EdgeInfo& edge)
{
    if (DragDropModel* model = qobject_cast<DragDropModel*>(this->model())) {
        for (int i = 0; i < model->rowCount(); i++) {
            if (model->index(i, m_allowDragColumn).data(Qt::DisplayRole).toString() == QString::fromStdString(edge.outNode + ":" + edge.outParam)) {
                model->removeRows(i, 1);
                break;
            }
        }
    }
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
        if (DragDropModel* dragModel = qobject_cast<DragDropModel*>(model())) {
            emit linksUpdated(dragModel->linksNeedUpdate());
        }
    }
    else {
        event->ignore();
    }
}

void ZenoDictListLinksTable::slt_clicked(const QModelIndex& index)
{
    QAbstractItemModel* m_model = model();
    int row = index.row();
    int column = index.column();
    if (column == 0) {
        if (row != 0) {
            QString uptext = m_model->data(m_model->index(row - 1, m_allowDragColumn), Qt::DisplayRole).toString();
            m_model->setData(m_model->index(row - 1, m_allowDragColumn), m_model->data(m_model->index(row, m_allowDragColumn), Qt::DisplayRole).toString(), Qt::DisplayRole);
            m_model->setData(m_model->index(row, m_allowDragColumn), uptext, Qt::DisplayRole);
            if (DragDropModel* dragModel = qobject_cast<DragDropModel*>(model())) {
                emit linksUpdated(dragModel->linksNeedUpdate());
            }
        }
    }
    else if (column == m_model->columnCount() - 1) {
        m_model->removeRows(row, 1);
        if (DragDropModel* dragModel = qobject_cast<DragDropModel*>(model())) {
            emit linksRemoved(dragModel->linksNeedRemove());
        }
    }
}
