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

#include "tree_item.h"

TreeItem::TreeItem()
    : _itemData()
    , _parentItem(nullptr)
{}

TreeItem::TreeItem(const QVariant& data)
    : _itemData(data)
    , _parentItem(nullptr)
{}

TreeItem::~TreeItem()
{
    qDeleteAll(_childItems);
}

TreeItem* TreeItem::parentItem()
{
    return _parentItem;
}

void TreeItem::setParentItem(TreeItem* parentItem)
{
    _parentItem = parentItem;
}

void TreeItem::appendChild(TreeItem* item)
{
    if (item && !_childItems.contains(item)) {
        _childItems.append(item);
    }
}

void TreeItem::removeChild(TreeItem* item)
{
    if (item) {
        _childItems.removeAll(item);
    }
}

TreeItem* TreeItem::child(int row)
{
    return _childItems.value(row);
}

int TreeItem::childCount() const
{
    return _childItems.count();
}

const QVariant& TreeItem::data() const
{
    return _itemData;
}

void TreeItem::setData(const QVariant& data)
{
    _itemData = data;
}

bool TreeItem::isLeaf() const
{
    return _childItems.isEmpty();
}

int TreeItem::depth() const
{
    int depth = 0;
    TreeItem* anchestor = _parentItem;
    while (anchestor) {
        ++depth;
        anchestor = anchestor->parentItem();
    }

    return depth;
}

int TreeItem::row() const
{
    if (_parentItem) {
        return _parentItem->_childItems.indexOf(const_cast<TreeItem*>(this));
    }

    return 0;
}
