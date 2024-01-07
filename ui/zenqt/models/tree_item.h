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

#ifndef QML_TREEVIEW_TREE_ITEM_H
#define QML_TREEVIEW_TREE_ITEM_H

#include <QVariant>

/*!
 * This class represents a node of the TreeModel.
 * The items are meant to be managed from the TreeModel, thus is only allowed
 * to modify the stored data.
 * Parenting and deletion are dealt from the TreeModel. Deleting a TreeItem
 * will call the delete for each child node.
 */
class TreeItem
{
    friend class TreeModel;

public:
    //! Create an empty item.
    TreeItem();

    //! Create an item with the given data.
    explicit TreeItem(const QVariant& data);

    //! Destroy the item. It will destroy every child.
    ~TreeItem();

    //! Return the stored data of the node.
    const QVariant& data() const;

    //! Set the internal data of the node.
    void setData(const QVariant& data);

    //! Return the number of children nodes.
    int childCount() const;

    int row() const;

    //! Return true if the node is a leaf node (no children).
    bool isLeaf() const;

    //! Return the depth of this node inside the tree.
    int depth() const;

private:
    TreeItem* parentItem();
    void setParentItem(TreeItem* parentItem);

    void appendChild(TreeItem* item);
    void removeChild(TreeItem* item);

    TreeItem* child(int row);

private:
    QVariant _itemData;
    TreeItem* _parentItem;
    QVector<TreeItem*> _childItems;
};

#endif // QML_TREEVIEW_TREE_ITEM_H
