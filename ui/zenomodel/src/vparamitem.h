#ifndef __VPARAM_ITEM_H__
#define __VPARAM_ITEM_H__

#include <QtWidgets>
#include <zenomodel/include/jsonhelper.h>
#include <rapidxml/rapidxml_print.hpp>
#include <zenomodel/include/modeldata.h>

using namespace rapidxml;

struct VParamItem;
class IGraphsModel;

class ProxySlotObject : public QObject
{
    Q_OBJECT
public:
    ProxySlotObject(VParamItem* pItem, QObject* parent = nullptr);
    ~ProxySlotObject();
    void mapCoreIndex(const QPersistentModelIndex& idx);
    void unmap();

public slots:
    void onDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles);

private:
    VParamItem* m_pItem;
};

struct VParamItem : public QStandardItem
{
    QPersistentModelIndex m_index;      //index to node param, see IParamModel.

    //for easy to debug, store here rather than QStandardItem internal data:
    QString m_name;

    PARAM_CONTROL m_ctrl;
    VPARAM_TYPE vType;
    VPARAM_INFO m_tempInfo;
    PARAM_LINKS m_links;
    SOCKET_PROPERTY m_sockProp;
    uint m_uuid;

    ProxySlotObject m_proxySlot;    //todo: pointer.

    QMap<int, QVariant> m_customData;

    VParamItem(VPARAM_TYPE vType, const QString& text, bool bMapCore = false);
    VParamItem(VPARAM_TYPE vType, const QIcon& icon, const QString& text, bool bMapCore = false);

    VParamItem(const VParamItem& other);
    ~VParamItem();

    QVariant data(int role = Qt::UserRole + 1) const override;
    void setData(const QVariant& value, int role) override;
    QStandardItem* clone() const override;
    void cloneFrom(VParamItem* pOther);
    void mapCoreParam(const QPersistentModelIndex& idx);
    rapidxml::xml_node<>* exportXml(rapidxml::xml_document<>& doc);
    VParamItem* getItem(const QString& uniqueName, int* r = 0) const;
    VParamItem* findItem(uint uuid, int* r = 0) const;
    VPARAM_INFO exportParamInfo();
    PARAM_CLASS getParamClass();
    void importParamInfo(const VPARAM_INFO& paramInfo);
    bool operator==(VParamItem* rItem) const;
    void read(QDataStream& in) override;
    void write(QDataStream& out) const override;

private:
    QVariant m_value;
    QString m_type;
};




#endif