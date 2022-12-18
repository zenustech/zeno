#ifndef __DICTKEY_MODEL_H__
#define __DICTKEY_MODEL_H__

#include <QtWidgets>

class DictKeyModel : public QStandardItemModel
{
    Q_OBJECT
public:
    DictKeyModel(QObject* parent = nullptr);
    bool moveRows(const QModelIndex &sourceParent, int sourceRow, int count,
                  const QModelIndex &destinationParent, int destinationChild) override;
};

#endif