/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2021 Maurizio Ingrassia
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "tree_model.h"

TreeModel::TreeModel(QObject* parent)
    : QAbstractItemModel(parent)
    , _rootItem{new TreeItem()}
{}

TreeModel::~TreeModel()
{
    delete _rootItem;
}

int TreeModel::rowCount(const QModelIndex& parent) const
{
    if (!parent.isValid()) {
        return _rootItem->childCount();
    }

    return internalPointer(parent)->childCount();
}

int TreeModel::columnCount(const QModelIndex& /*parent*/) const
{
    // This is basically flatten as a list model
    return 1;
}

QModelIndex TreeModel::index(const int row, const int column, const QModelIndex& parent) const
{
    if (!hasIndex(row, column, parent)) {
        return {};
    }

    TreeItem* item = _rootItem;
    if (parent.isValid()) {
        item = internalPointer(parent);
    }

    if (auto child = item->child(row)) {
        return createIndex(row, column, child);
    }

    return {};
}

QModelIndex TreeModel::parent(const QModelIndex& index) const
{
    if (!index.isValid()) {
        return {};
    }

    TreeItem* childItem = internalPointer(index);
    TreeItem* parentItem = childItem->parentItem();

    if (!parentItem) {
        return {};
    }

    if (parentItem == _rootItem) {
        return {};
    }

    return createIndex(parentItem->row(), 0, parentItem);
}

QVariant TreeModel::data(const QModelIndex& index, const int role) const
{
    if (!index.isValid() || role != Qt::DisplayRole) {
        return QVariant();
    }

    return internalPointer(index)->data();
}

bool TreeModel::setData(const QModelIndex& index, const QVariant& value, int /*role*/)
{
    if (!index.isValid()) {
        return false;
    }

    if (auto item = internalPointer(index)) {
        item->setData(value);
        emit dataChanged(index, index, {Qt::EditRole});
    }

    return false;
}

void TreeModel::addTopLevelItem(TreeItem* child)
{
    if (child) {
        addItem(_rootItem, child);
    }
}

void TreeModel::addItem(TreeItem* parent, TreeItem* child)
{
    if (!child || !parent) {
        return;
    }

    emit layoutAboutToBeChanged();

    if (child->parentItem()) {
        beginRemoveRows(QModelIndex(), qMax(parent->childCount() - 1, 0), qMax(parent->childCount(), 0));
        child->parentItem()->removeChild(child);
        endRemoveRows();
    }

    beginInsertRows(QModelIndex(), qMax(parent->childCount() - 1, 0), qMax(parent->childCount() - 1, 0));
    child->setParentItem(parent);
    parent->appendChild(child);
    endInsertRows();

    emit layoutChanged();
}

void TreeModel::removeItem(TreeItem* item)
{
    if (!item) {
        return;
    }

    emit layoutAboutToBeChanged();

    if (item->parentItem()) {
        beginRemoveRows(QModelIndex(), qMax(item->childCount() - 1, 0), qMax(item->childCount(), 0));
        item->parentItem()->removeChild(item);
        endRemoveRows();
    }

    emit layoutChanged();
}

TreeItem* TreeModel::rootItem() const
{
    return _rootItem;
}

QModelIndex TreeModel::rootIndex()
{
    return {};
}

int TreeModel::depth(const QModelIndex& index) const
{
    int count = 0;
    auto anchestor = index;
    if (!index.isValid()) {
        return 0;
    }
    while (anchestor.parent().isValid()) {
        anchestor = anchestor.parent();
        ++count;
    }

    return count;
}

void TreeModel::clear()
{
    emit layoutAboutToBeChanged();
    beginResetModel();
    delete _rootItem;
    _rootItem = new TreeItem();
    endResetModel();
    emit layoutChanged();
}

TreeItem* TreeModel::internalPointer(const QModelIndex& index) const
{
    return static_cast<TreeItem*>(index.internalPointer());
}
