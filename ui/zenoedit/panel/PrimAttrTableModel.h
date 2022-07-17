//
// Created by zh on 2022/6/30.
//

#ifndef ZENO_PRIMATTRTABLEMODEL_H
#define ZENO_PRIMATTRTABLEMODEL_H

#include <QtWidgets>
#include "zeno/core/IObject.h"
#include "zeno/types/PrimitiveObject.h"

class PrimAttrTableModel : public QAbstractTableModel {
    Q_OBJECT
public:
    explicit PrimAttrTableModel(QObject *parent = 0);
    int rowCount(const QModelIndex &parent) const override;
    int columnCount(const QModelIndex &parent) const override;
    QVariant data(const QModelIndex &index, int role) const override;
//    bool	setData(const QModelIndex &index, const QVariant &value, int role = Qt::EditRole)
    QVariant headerData(int section, Qt::Orientation orientation, int role) const override;
    void setModelData(zeno::PrimitiveObject* prim);
    void setSelAttr(std::string sel_attr_);

private:
    std::shared_ptr<zeno::PrimitiveObject> m_prim = nullptr;
    std::string sel_attr = "Vertex";
};


#endif //ZENO_PRIMATTRTABLEMODEL_H
