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

#ifndef QML_TREEVIEW_TREE_MODEL_H
#define QML_TREEVIEW_TREE_MODEL_H

#include "tree_item.h"

#include <QAbstractItemModel>

/*!
 * The Tree Model works as List Model, using one column and using the row information
 * referred to the parent node.
 */
class TreeModel : public QAbstractItemModel
{
    Q_OBJECT

public:
    explicit TreeModel(QObject* parent = nullptr);
    ~TreeModel() override;

    // Overriden method from QAbstractItemModel

public:
    int rowCount(const QModelIndex& index) const override;
    int columnCount(const QModelIndex& index) const override;

    QModelIndex index(int row, int column, const QModelIndex& parent) const override;
    QModelIndex parent(const QModelIndex& childIndex) const override;

    QVariant data(const QModelIndex& index, int role = 0) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;

public:
    //! Add an item to the top level.
    void addTopLevelItem(TreeItem* child);

    //! Add a child to the parent item.
    void addItem(TreeItem* parent, TreeItem* child);

    //! Remove the item from the model.
    void removeItem(TreeItem* item);

    //! Return the root item of the model.
    TreeItem* rootItem() const;

    //! Return the depth for the given index
    Q_INVOKABLE int depth(const QModelIndex& index) const;

    //! Clear the model.
    Q_INVOKABLE void clear();

    /*!
    *  Return the root item to the QML Side.
    *  This method is not meant to be used in client code.
    */
    Q_INVOKABLE QModelIndex rootIndex();

private:
    TreeItem* internalPointer(const QModelIndex& index) const;

private:
    TreeItem* _rootItem;
};

#endif // QML_TREEVIEW_TREE_MODEL_H
