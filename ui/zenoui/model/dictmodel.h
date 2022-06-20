#ifndef __DICT_MODEL_H__
#define __DICT_MODEL_H__

#include <QStandardItemModel>

enum DICT_ROLE {
    ROLE_KEY = Qt::UserRole + 1,
    ROLE_DATATYPE,
    ROLE_VALUE,
};

class DictModel : public QStandardItemModel
{
    Q_OBJECT
public:
    explicit DictModel(QObject* parent = nullptr);
    //int columnCount(const QModelIndex &parent = QModelIndex()) const override;
    //QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;

private:

};

#endif