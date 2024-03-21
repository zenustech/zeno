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
    void setStrMapping(bool enable);

    zeno::zany userDataByIndex(const QModelIndex& index) const;

private:
    std::shared_ptr<zeno::PrimitiveObject> m_prim = nullptr;
    std::string sel_attr = "Vertex";
    bool enable_str_mapping = false;

    QVariant vertexData(const QModelIndex &index) const;
    QVariant trisData(const QModelIndex &index) const;
    QVariant pointsData(const QModelIndex &index) const;
    QVariant linesData(const QModelIndex &index) const;
    QVariant quadsData(const QModelIndex &index) const;
    QVariant polysData(const QModelIndex &index) const;
    QVariant loopsData(const QModelIndex &index) const;
    QVariant loopUVsData(const QModelIndex &index) const;
    QVariant uvsData(const QModelIndex &index) const;
    QVariant userData(const zeno::zany& object) const;
};


#endif //ZENO_PRIMATTRTABLEMODEL_H
