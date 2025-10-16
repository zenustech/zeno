#include "customuimodel.h"
#include <zeno/utils/helper.h>
#include <zeno/core/typeinfo.h>
#include <unordered_set>
#include <string>
#include "style/colormanager.h"
#include "util/uihelper.h"
#include "zeno_types/reflect/reflection.generated.hpp"
#include "declmetatype.h"


using namespace zeno::types;


static CUSTOMUI_CTRL_ITEM_INFO customui_controlList[] = {
    {"Tab",                 zeno::NullControl,  Param_Null,   "qrc:/icons/parameter_control_tab.svg"},
    {"Group",               zeno::NullControl,  Param_Null,   "qrc:/icons/parameter_control_group.svg"},
    {"object",              zeno::NullControl,  Param_Null,   ""},
    {"Integer",             zeno::Lineedit,     gParamType_Int,    "qrc:/icons/parameter_control_integer.svg"},
    {"Float",               zeno::Lineedit,     gParamType_Float,  "qrc:/icons/parameter_control_float.svg"},
    {"String",              zeno::Lineedit,     gParamType_String, "qrc:/icons/parameter_control_string.svg"},
    {"Boolean",             zeno::Checkbox,     gParamType_Bool,   "qrc:/icons/parameter_control_boolean.svg"},
    {"Multiline String",    zeno::Multiline,    gParamType_String, "qrc:/icons/parameter_control_string.svg"},
    {"read path",           zeno::ReadPathEdit, gParamType_String, "qrc:/icons/parameter_control_fold.svg"},
    {"write path",          zeno::WritePathEdit,gParamType_String, "qrc:/icons/parameter_control_fold.svg"},
    {"directory",       zeno::DirectoryPathEdit,gParamType_String, "qrc:/icons/parameter_control_fold.svg"},
    {"Enum",                zeno::Combobox,     gParamType_String, "qrc:/icons/parameter_control_enum.svg"},
    {"Float Vector 4",      zeno::Vec4edit,     gParamType_Vec4f,  "qrc:/icons/parameter_control_floatVector4.svg"},
    {"Float Vector 3",      zeno::Vec3edit,     gParamType_Vec3f,  "qrc:/icons/parameter_control_floatVector3.svg"},
    {"Float Vector 2",      zeno::Vec2edit,     gParamType_Vec2f,  "qrc:/icons/parameter_control_floatVector2.svg"},
    {"Integer Vector 4",    zeno::Vec4edit,     gParamType_Vec4i,  "qrc:/icons/parameter_control_integerVector4.svg"},
    {"Integer Vector 3",    zeno::Vec3edit,     gParamType_Vec3i,  "qrc:/icons/parameter_control_integerVector3.svg"},
    {"Integer Vector 2",    zeno::Vec2edit,     gParamType_Vec2i,  "qrc:/icons/parameter_control_integerVector2.svg"},
    {"Color",               zeno::Heatmap,      gParamType_Heatmap,"qrc:/icons/parameter_control_color.svg"},
    {"Color Vec3f",         zeno::ColorVec,     gParamType_Vec3f,  "qrc:/icons/parameter_control_color.svg"},
    {"Curve",               zeno::CurveEditor,  gParamType_Curve,  "qrc:/icons/parameter_control_curve.svg"},
    {"SpinBox",             zeno::SpinBox,      gParamType_Int,    "qrc:/icons/parameter_control_spinbox.svg"},
    {"DoubleSpinBox",       zeno::DoubleSpinBox,gParamType_Float,  "qrc:/icons/parameter_control_spinbox.svg"},
    {"Slider",              zeno::Slider,       gParamType_Int,    "qrc:/icons/parameter_control_slider.svg"},
    {"SpinBoxSlider",       zeno::SpinBoxSlider,gParamType_Int,    "qrc:/icons/parameter_control_slider.svg"},
};

static bool hasElement(QAbstractListModel* pModel, const QString& name) {
    for (int r = 0; r < pModel->rowCount(); r++) {
        QModelIndex idx = pModel->index(r, 0);
        if (idx.data(QtRole::ROLE_PARAM_NAME) == name)
            return true;
    }
    return false;
}

CustomUIModel::CustomUIModel(ParamsModel* params, QObject* parent)
    : QObject(parent)
    , m_params(params)
    , m_tabModel(nullptr)
    , m_primOutputModel(nullptr)
    , m_objInputModel(nullptr)
    , m_objOutputModel(nullptr)
    , m_ctrlItemModel(nullptr)
{
    const zeno::CustomUI& customUI = m_params->customUI();
    m_tabModel = new ParamTabModel(customUI.inputPrims, this);
    m_primOutputModel = new PrimParamOutputModel(customUI.outputPrims, this);
    m_objInputModel = new objParamInputModel(customUI.inputObjs, this);
    m_objOutputModel = new objParamOutputModel(customUI.outputObjs, this);
    m_ctrlItemModel = new ControlItemListModel(this);
}

ParamTabModel* CustomUIModel::tabModel() const
{
    return m_tabModel;
}

PrimParamOutputModel* CustomUIModel::primOutputModel() const
{
    return m_primOutputModel;
}

objParamInputModel* CustomUIModel::objInputModel() const
{
    return m_objInputModel;
}

objParamOutputModel* CustomUIModel::objOutputModel() const
{
    return m_objOutputModel;
}

ParamsModel* CustomUIModel::coreModel() const {
    return m_params;
}

ControlItemListModel* CustomUIModel::controlItemModel() const
{
    return m_ctrlItemModel;
}

void CustomUIModel::initCustomuiConnections(QStandardItemModel* customuiStandarditemModel)
{
    connect(customuiStandarditemModel, &QStandardItemModel::rowsInserted, [customuiStandarditemModel, this](const QModelIndex& parent, int first, int last) {
        QStandardItem* parentItem = customuiStandarditemModel->itemFromIndex(parent);
        QStandardItem* newItem = parentItem->child(first);
        int elemtype = newItem->data(ROLE_ELEMENT_TYPE).toInt();
        ParamTabModel* tabmodel = tabModel();
        if (elemtype == VPARAM_TAB) {
            tabmodel->insertRow(first, newItem->data(QtRole::ROLE_PARAM_NAME).toString());
        } 
        else if (elemtype == VPARAM_GROUP) {
            ParamGroupModel* group = tabmodel->index(parent.row()).data(QmlCUIRole::GroupModel).value<ParamGroupModel*>();
            group->insertRow(first, newItem->data(QtRole::ROLE_PARAM_NAME).toString());
        }
        else if (elemtype == VPARAM_PARAM) {
            const QModelIndex& tabindx = parent.parent();
            if (ParamGroupModel* groupmodel = tabmodel->index(tabindx.row()).data(QmlCUIRole::GroupModel).value<ParamGroupModel*>()) {
                ParamPlainModel* paramsmodel = groupmodel->index(parent.row()).data(QmlCUIRole::PrimModel).value<ParamPlainModel*>();
                //paramsmodel->insertRow(first, newItem->data(QtRole::ROLE_PARAM_NAME).toString());
            }
        }
    });
    connect(customuiStandarditemModel, &QStandardItemModel::rowsAboutToBeRemoved, [customuiStandarditemModel, this](const QModelIndex& parent, int first, int last) {
        QStandardItem* parentItem = customuiStandarditemModel->itemFromIndex(parent);
        QStandardItem* removeItem = parentItem->child(first);
        int elemtype = removeItem->data(ROLE_ELEMENT_TYPE).toInt();
        ParamTabModel* tabmodel = tabModel();
        if (elemtype == VPARAM_TAB) {
            tabmodel->removeRow(first);
        }
        else if (elemtype == VPARAM_GROUP) {
            ParamGroupModel* group = tabmodel->index(parent.row()).data(QmlCUIRole::GroupModel).value<ParamGroupModel*>();
            group->removeRow(first);
        }
        else if (elemtype == VPARAM_PARAM) {
            const QModelIndex& tabindx = parent.parent();
            if (ParamGroupModel* groupmodel = tabmodel->index(tabindx.row()).data(QmlCUIRole::GroupModel).value<ParamGroupModel*>()) {
                ParamPlainModel* paramsmodel = groupmodel->index(parent.row()).data(QmlCUIRole::PrimModel).value<ParamPlainModel*>();
                paramsmodel->removeRow(first);

                tabmodel->setData(tabmodel->index(0,0), "dddd", QtRole::ROLE_PARAM_NAME);
            }
        }
    });
    connect(customuiStandarditemModel, &QStandardItemModel::dataChanged, [customuiStandarditemModel, this](const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles) {
        if (roles.size() == 1 && roles[0] == QtRole::ROLE_PARAM_NAME) {
            QStandardItem* changeItem = customuiStandarditemModel->itemFromIndex(topLeft);
            int elemtype = changeItem->data(ROLE_ELEMENT_TYPE).toInt();
            ParamTabModel* tabmodel = tabModel();
            if (elemtype == VPARAM_TAB) {
                tabmodel->setData(topLeft, topLeft.data(QtRole::ROLE_PARAM_NAME), QtRole::ROLE_PARAM_NAME);
            } else if (elemtype == VPARAM_GROUP) {
                ParamGroupModel* group = tabmodel->index(topLeft.parent().row()).data(QmlCUIRole::GroupModel).value<ParamGroupModel*>();
                group->setData(topLeft, topLeft.data(QtRole::ROLE_PARAM_NAME), QtRole::ROLE_PARAM_NAME);
            } else if (elemtype == VPARAM_PARAM) {
                const QModelIndex& tabindx = topLeft.parent().parent();
                if (ParamGroupModel* groupmodel = tabmodel->index(tabindx.row()).data(QmlCUIRole::GroupModel).value<ParamGroupModel*>()) {
                    ParamPlainModel* paramsmodel = groupmodel->index(topLeft.parent().row()).data(QmlCUIRole::PrimModel).value<ParamPlainModel*>();
                    paramsmodel->setData(topLeft, topLeft.data(QtRole::ROLE_PARAM_NAME), QtRole::ROLE_PARAM_NAME);
                }
            }
        }
    });
}

Q_INVOKABLE void CustomUIModel::reset()
{
    m_tabModel->reset();
    m_primOutputModel->reset();
    m_objInputModel->reset();
    m_objOutputModel->reset();
    emit resetCustomuiModel();
}

bool CustomUIModel::isSubnetNode()
{
    if (m_params->rowCount() != 0) {
        auto idx = m_params->index(0);
        QModelIndex nodeidx = m_params->data(idx, QtRole::ROLE_NODEIDX).value<QModelIndex>();
        if (nodeidx.isValid() && (zeno::NodeType)nodeidx.data(QtRole::ROLE_NODETYPE).toInt() == zeno::Node_SubgraphNode) {
            return true;
        }
    }
    return false;
}

//////////////////////////////////////////////////////////
ParamTabModel::ParamTabModel(zeno::CustomUIParams tabs, CustomUIModel* pModel)
    : QAbstractListModel(pModel)
    , m_customuiM(pModel)
{
    for (const zeno::ParamTab& tab : tabs) {
        _TabItem item;
        item.name = QString::fromStdString(tab.name);
        item.groupM = new ParamGroupModel(tab, this);
        m_items.push_back(std::move(item));
    }
}

QStandardItemModel* ParamTabModel::toStandardModel() const {
	QStandardItem* rootItem = new QStandardItem("Root");
    rootItem->setData(VPARAM_ROOT, ROLE_ELEMENT_TYPE);

    QStandardItemModel* pModel = new QStandardItemModel;
    for (auto& tab : m_items) {
        QStandardItem* tabItem = new QStandardItem(tab.name);
        tabItem->setData(VPARAM_TAB, ROLE_ELEMENT_TYPE);
        tabItem->setData(zeno::Role_InputPrimitive, QtRole::ROLE_PARAM_GROUP);
        tabItem->setData(tab.name, QtRole::ROLE_PARAM_NAME);

        ParamGroupModel* groupsM = tab.groupM;
        for (int i = 0; i < groupsM->rowCount(); i++) {
            QModelIndex groupIdx = groupsM->index(i);
            QString groupName = groupsM->data(groupIdx, Qt::DisplayRole).toString();
            ParamPlainModel* paramsM = groupsM->data(groupIdx, QmlCUIRole::PrimModel).value<ParamPlainModel*>();
            QStandardItem* groupItem = new QStandardItem(groupName);
            groupItem->setData(VPARAM_GROUP, ROLE_ELEMENT_TYPE);
            groupItem->setData(zeno::Role_InputPrimitive, QtRole::ROLE_PARAM_GROUP);
            groupItem->setData(groupName, QtRole::ROLE_PARAM_NAME);

            for (int j = 0; j < paramsM->rowCount(); j++) {
                QModelIndex idx = paramsM->index(j);
                auto paramName = idx.data(QtRole::ROLE_PARAM_NAME).toString();
                QStandardItem* pItem = new QStandardItem(paramName);
                pItem->setData(paramName, QtRole::ROLE_PARAM_NAME);
                pItem->setData(paramName, QtRole::ROLE_PARAM_NAME_EXIST);
                pItem->setData(idx.data(QtRole::ROLE_PARAM_TYPE), QtRole::ROLE_PARAM_TYPE);
                pItem->setData(idx.data(QtRole::ROLE_PARAM_VALUE), QtRole::ROLE_PARAM_VALUE);
                pItem->setData(idx.data(QtRole::ROLE_PARAM_CONTROL), QtRole::ROLE_PARAM_CONTROL);
                pItem->setData(idx.data(QtRole::ROLE_SOCKET_TYPE), QtRole::ROLE_SOCKET_TYPE);
                pItem->setData(idx.data(QtRole::ROLE_ISINPUT), QtRole::ROLE_ISINPUT);
                pItem->setData(idx.data(QtRole::ROLE_NODEIDX), QtRole::ROLE_NODEIDX);
                pItem->setData(idx.data(QtRole::ROLE_PARAM_CTRL_PROPERTIES), QtRole::ROLE_PARAM_CTRL_PROPERTIES);
                pItem->setData(idx.data(QtRole::ROLE_PARAM_CONTROL_PROPS), QtRole::ROLE_PARAM_CONTROL_PROPS);
                pItem->setData(idx.data(QtRole::ROLE_PARAM_VISIBLE), QtRole::ROLE_PARAM_VISIBLE);
                pItem->setData(idx.data(QtRole::ROLE_PARAM_SOCKET_VISIBLE), QtRole::ROLE_PARAM_SOCKET_VISIBLE);
                pItem->setData(idx.data(QtRole::ROLE_PARAM_GROUP), QtRole::ROLE_PARAM_GROUP);
                pItem->setData(VPARAM_PARAM, ROLE_ELEMENT_TYPE);
                groupItem->appendRow(pItem);
            }
            tabItem->appendRow(groupItem);
        }
        rootItem->appendRow(tabItem);
    }
    pModel->invisibleRootItem()->appendRow(rootItem);
    return pModel;
}

int ParamTabModel::rowCount(const QModelIndex& parent) const {
    return m_items.size();
}

QVariant ParamTabModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid()) {
        return false;
    }
    if (role == QtRole::ROLE_PARAM_NAME) {
        return m_items[index.row()].name;
    }
    else if (role == QmlCUIRole::GroupModel) {
        return QVariant::fromValue(m_items[index.row()].groupM);
    }
    else {
        return QVariant();
    }
}

bool ParamTabModel::setData(const QModelIndex& index, const QVariant& value, int role) {
    if (!index.isValid()) {
        return false;
    }
    switch (role) {
    case QtRole::ROLE_PARAM_NAME:
        m_items[index.row()].name = value.toString();
        emit dataChanged(index, index, { QtRole::ROLE_PARAM_NAME });
        return true;
    }
    return false;
}

QHash<int, QByteArray> ParamTabModel::roleNames() const {
    QHash<int, QByteArray> values;
    values[QtRole::ROLE_PARAM_NAME] = "tabname";
    values[QmlCUIRole::GroupModel] = "groups";
    return values;
}

Qt::ItemFlags ParamTabModel::flags(const QModelIndex& index) const
{
    if (!index.isValid())
        return Qt::NoItemFlags;
    return Qt::ItemIsEditable | QAbstractListModel::flags(index);
}

QModelIndex ParamTabModel::indexFromName(QString name)
{
	for (int i = 0; i < m_items.size(); i++) {
		if (m_items[i].name == name) {
			return createIndex(i, 0);
		}
	}
	return QModelIndex();
}

bool ParamTabModel::insertRow(int row, QString name) {
    if (row < 0 || row > m_items.size()) {
        return false;
    }

    beginInsertRows(QModelIndex(), row, row);
    _TabItem item;
    item.name = name;
    item.groupM = new ParamGroupModel(zeno::ParamTab{name.toStdString()}, this);
    m_items.insert(row, std::move(item));
    endInsertRows();
    return true;
}

bool ParamTabModel::removeRow(int row)
{
    if (row < 0 || row >= m_items.size()) {
        return false;
    }

    beginRemoveRows(QModelIndex(), row, row);
    m_items.erase(m_items.begin() + row);
    endRemoveRows();
    return true;
}

void ParamTabModel::reset()
{
    beginResetModel();
    for (auto& item : m_items) {
        delete item.groupM;
    }
    m_items.clear();
    const zeno::CustomUI& customUI = m_customuiM->coreModel()->customUI();
    for (const zeno::ParamTab& tab : customUI.inputPrims) {
        _TabItem item;
        item.name = QString::fromStdString(tab.name);
        item.groupM = new ParamGroupModel(tab, this);
        m_items.push_back(std::move(item));
    }
    endResetModel();
}

//////////////////////////////////////////////////
ParamGroupModel::ParamGroupModel(zeno::ParamTab tab, ParamTabModel* pModel)
    : QAbstractListModel(pModel)
    , m_tabModel(pModel)
{
    for (const zeno::ParamGroup& group : tab.groups) {
        _GroupItem item;
        item.name = QString::fromStdString(group.name);
        item.paramM = new ParamPlainModel(group, this);
        m_items.push_back(std::move(item));
    }
}

int ParamGroupModel::rowCount(const QModelIndex& parent) const {
    return m_items.size();
}

QVariant ParamGroupModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid()) {
        return false;
    }
    if (role == Qt::DisplayRole || role == QtRole::ROLE_PARAM_NAME) {
        return m_items[index.row()].name;
    }
    else if (role == QmlCUIRole::PrimModel) {
        return QVariant::fromValue(m_items[index.row()].paramM);
    }
    return QVariant();
}

bool ParamGroupModel::setData(const QModelIndex& index, const QVariant& value, int role) {
    if (!index.isValid()) {
        return false;
    }
    switch (role) {
    case Qt::DisplayRole:
        m_items[index.row()].name = value.toString();
        emit dataChanged(index, index, { Qt::DisplayRole });
        return true;
    }
    return false;
}

QHash<int, QByteArray> ParamGroupModel::roleNames() const {
    QHash<int, QByteArray> values;
    values[Qt::DisplayRole] = "groupname";
    values[QmlCUIRole::PrimModel] = "params";
    return values;
}


QModelIndex ParamGroupModel::indexFromName(QString name)
{
    for (int i = 0; i < m_items.size(); i++) {
        if (m_items[i].name == name) {
            return createIndex(i, 0);
        }
    }
    return QModelIndex();
}

QString ParamGroupModel::parentName()
{
    if (!m_tabModel) {
        return "";
    }
    for (int i = 0; i < m_tabModel->rowCount(); i++) {
        ParamGroupModel* pgroup = m_tabModel->data(m_tabModel->index(i), QmlCUIRole::GroupModel).value<ParamGroupModel*>();
        if (this == pgroup) {
            return m_tabModel->data(m_tabModel->index(i), QtRole::ROLE_PARAM_NAME).toString();
        }
    }
    return "";
}

bool ParamGroupModel::insertRow(int row, QString name)
{
    if (row < 0 || row > m_items.size()) {
        return false;
    }

    beginInsertRows(QModelIndex(), row, row);
    _GroupItem item;
    item.name = name;
    item.paramM = new ParamPlainModel(zeno::ParamGroup{ name.toStdString() }, this);
    m_items.insert(row, std::move(item));
    endInsertRows();
    return true;
}

bool ParamGroupModel::removeRow(int row)
{
    if (row < 0 || row >= m_items.size()) {
        return false;
    }

    beginRemoveRows(QModelIndex(), row, row);
    m_items.erase(m_items.begin() + row);
    endRemoveRows();
    return true;
}

////////////////////////////////////////////////////////
ParamPlainModel::ParamPlainModel(zeno::ParamGroup group, ParamGroupModel* pModel)
    : QAbstractListModel(pModel)
    , m_groupModel(pModel)
    , m_paramsModel(nullptr)
{
    m_paramsModel = m_groupModel->tabModel()->uimodel()->coreModel();
    for (const zeno::ParamPrimitive& param : group.params) {
        int r = m_paramsModel->indexFromName(QString::fromStdString(param.name), true);
        QPersistentModelIndex idx = m_paramsModel->index(r);
        m_items.push_back(idx);
    }
}

QString ParamPlainModel::getMaxLengthName() const
{
    QString maxName;
	for (auto& idx : m_items) {
		const QString& name = idx.data(QtRole::ROLE_PARAM_NAME).toString();
		if (name.length() > maxName.length()) {
			maxName = name;
		}
	}
    return maxName;
}

int ParamPlainModel::rowCount(const QModelIndex& parent) const {
    return m_items.size();
}

QVariant ParamPlainModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid()) {
        return QVariant();
    }
    if (role == Qt::DecorationRole) {
        auto currtype = m_items[index.row()].data(QtRole::ROLE_PARAM_TYPE).toLongLong();
        for (auto& item: customui_controlList) {
            if (item.type == currtype) {
                return item.icon;
            }
        }
    }
    return m_items[index.row()].data(role);
}

bool ParamPlainModel::setData(const QModelIndex& index, const QVariant& value, int role) {
    return m_paramsModel->setData(m_items[index.row()], value, role);
}

QHash<int, QByteArray> ParamPlainModel::roleNames() const {
    //copy from ParamsModel::roleNames()
    QHash<int, QByteArray> roles;
    roles[QtRole::ROLE_NODE_NAME] = "nodename";
    roles[QtRole::ROLE_PARAM_NAME] = "name";
    roles[QtRole::ROLE_PARAM_TYPE] = "type";
    roles[QtRole::ROLE_PARAM_CONTROL] = "control";
    roles[QtRole::ROLE_PARAM_QML_VALUE] = "value";
    roles[QtRole::ROLE_ISINPUT] = "input";
    roles[QtRole::ROLE_PARAM_GROUP] = "group";
    roles[QtRole::ROLE_PARAM_SOCKET_VISIBLE] = "socket_visible";
    roles[QtRole::ROLE_PARAM_CONTROL_PROPS] = "control_properties";
    roles[QtRole::ROLE_PARAM_SOCKET_CLR] = "socket_color";
    roles[QtRole::ROLE_PARAM_PERSISTENT_INDEX] = "per_index";
    roles[QtRole::ROLE_PARAM_VISIBLE] = "param_visible";
    roles[Qt::DecorationRole] = "paramIcon";
    return roles;
}


QString ParamPlainModel::parentName()
{
    if (!m_groupModel) {
        return "";
    }
    for (int i = 0; i < m_groupModel->rowCount(); i++) {
        ParamPlainModel* plainM = m_groupModel->data(m_groupModel->index(i), QmlCUIRole::PrimModel).value<ParamPlainModel*>();
        if (this == plainM) {
            return m_groupModel->data(m_groupModel->index(i), QtRole::ROLE_PARAM_NAME).toString();
        }
    }
    return "";
}

QPersistentModelIndex ParamPlainModel::getParamsModelIndex(int row)
{
    if (row >= m_items.size()) {
        return QPersistentModelIndex();
    }
    return m_items[row];
}

bool ParamPlainModel::insertRow(int row, QString name, int ctrlItemRow)
{
    return false;
}

bool ParamPlainModel::removeRow(int row)
{
    return false;
}

///////////////////////////////////////////////////////////////
PrimParamOutputModel::PrimParamOutputModel(zeno::PrimitiveParams params, CustomUIModel* pModel)
    : QAbstractListModel(pModel)
    , m_paramsModel(nullptr)
{
    m_paramsModel = pModel->coreModel();
    for (const zeno::ParamPrimitive& param : params) {
        int r = m_paramsModel->indexFromName(QString::fromStdString(param.name), false);
        QPersistentModelIndex idx = m_paramsModel->index(r);
        m_items.push_back(idx);
    }
}

int PrimParamOutputModel::rowCount(const QModelIndex& parent) const {
    return m_items.size();
}

QVariant PrimParamOutputModel::data(const QModelIndex& index, int role) const {
    return m_items[index.row()].data(role);
}

bool PrimParamOutputModel::setData(const QModelIndex& index, const QVariant& value, int role) {
    return m_paramsModel->setData(m_items[index.row()], value, role);
}

QHash<int, QByteArray> PrimParamOutputModel::roleNames() const {
    //copy from ParamsModel::roleNames()
    QHash<int, QByteArray> roles;
    roles[QtRole::ROLE_NODE_NAME] = "nodename";
    roles[QtRole::ROLE_PARAM_NAME] = "name";
    roles[QtRole::ROLE_PARAM_TYPE] = "type";
    roles[QtRole::ROLE_PARAM_CONTROL] = "control";
    roles[QtRole::ROLE_ISINPUT] = "input";
    roles[QtRole::ROLE_PARAM_GROUP] = "group";
    roles[QtRole::ROLE_PARAM_SOCKET_VISIBLE] = "socket_visible";
    roles[QtRole::ROLE_PARAM_CONTROL_PROPS] = "control_properties";
    roles[QtRole::ROLE_PARAM_SOCKET_CLR] = "socket_color";
    roles[QtRole::ROLE_PARAM_VISIBLE] = "param_visible";
    return roles;
}

QStandardItemModel* PrimParamOutputModel::toStandardModel() const {
    QStandardItemModel* pModel = new QStandardItemModel;
    for (int j = 0; j < rowCount(); j++) {
        QModelIndex idx = index(j);
        auto paramName = idx.data(QtRole::ROLE_PARAM_NAME).toString();
        QStandardItem* pItem = new QStandardItem(paramName);
        pItem->setData(paramName, QtRole::ROLE_PARAM_NAME);
        pItem->setData(paramName, QtRole::ROLE_PARAM_NAME_EXIST);
        pItem->setData(idx.data(QtRole::ROLE_PARAM_TYPE), QtRole::ROLE_PARAM_TYPE);
        pItem->setData(idx.data(QtRole::ROLE_PARAM_VALUE), QtRole::ROLE_PARAM_VALUE);
        pItem->setData(idx.data(QtRole::ROLE_PARAM_CONTROL), QtRole::ROLE_PARAM_CONTROL);
        pItem->setData(idx.data(QtRole::ROLE_SOCKET_TYPE), QtRole::ROLE_SOCKET_TYPE);
        pItem->setData(idx.data(QtRole::ROLE_ISINPUT), QtRole::ROLE_ISINPUT);
        pItem->setData(idx.data(QtRole::ROLE_NODEIDX), QtRole::ROLE_NODEIDX);
        pItem->setData(idx.data(QtRole::ROLE_PARAM_CONTROL_PROPS), QtRole::ROLE_PARAM_CONTROL_PROPS);
        pItem->setData(idx.data(QtRole::ROLE_PARAM_VISIBLE), QtRole::ROLE_PARAM_VISIBLE);
        pItem->setData(idx.data(QtRole::ROLE_PARAM_SOCKET_VISIBLE), QtRole::ROLE_PARAM_SOCKET_VISIBLE);
        pItem->setData(idx.data(QtRole::ROLE_PARAM_GROUP), QtRole::ROLE_PARAM_GROUP);
        pItem->setData(VPARAM_PARAM, ROLE_ELEMENT_TYPE);
        pModel->appendRow(pItem);
    }
    return pModel;
}

QPersistentModelIndex PrimParamOutputModel::getParamsModelIndex(int row)
{
	if (row >= m_items.size()) {
		return QPersistentModelIndex();
	}
	return m_items[row];
}

QString PrimParamOutputModel::getMaxLengthName() const
{
    QString maxName;
    for (auto& idx : m_items) {
        const QString& name = idx.data(QtRole::ROLE_PARAM_NAME).toString();
        if (name.length() > maxName.length()) {
            maxName = name;
        }
    }
    return maxName;
}

Q_INVOKABLE bool PrimParamOutputModel::insertRow(int row, QString name)
{
    return false;
}

Q_INVOKABLE bool PrimParamOutputModel::removeRow(int row)
{
    return false;
}

void PrimParamOutputModel::reset()
{
    beginResetModel();
    m_items.clear();
    const zeno::CustomUI& customUI = m_paramsModel->customUI();
    for (const zeno::ParamPrimitive& param : customUI.outputPrims) {
        int r = m_paramsModel->indexFromName(QString::fromStdString(param.name), false);
        QPersistentModelIndex idx = m_paramsModel->index(r);
        m_items.push_back(idx);
    }
    endResetModel();
}

objParamInputModel::objParamInputModel(zeno::ObjectParams params, CustomUIModel* pModel)
    : QAbstractListModel(pModel)
    , m_paramsModel(nullptr)
{
    m_paramsModel = pModel->coreModel();
    for (const zeno::ParamObject& param : params) {
        int r = m_paramsModel->indexFromName(QString::fromStdString(param.name), true);
        QPersistentModelIndex idx = m_paramsModel->index(r);
        m_items.push_back(idx);
    }
}


int objParamInputModel::rowCount(const QModelIndex& parent /*= QModelIndex()*/) const
{
    return m_items.size();
}

QVariant objParamInputModel::data(const QModelIndex& index, int role /*= Qt::DisplayRole*/) const
{
    return m_items[index.row()].data(role);
}

bool objParamInputModel::setData(const QModelIndex& index, const QVariant& value, int role /*= Qt::EditRole*/)
{
    return m_paramsModel->setData(m_items[index.row()], value, role);
}

QHash<int, QByteArray> objParamInputModel::roleNames() const
{
    QHash<int, QByteArray> roles;
    roles[QtRole::ROLE_NODE_NAME] = "nodename";
    roles[QtRole::ROLE_PARAM_NAME] = "paramname";
    roles[QtRole::ROLE_PARAM_TYPE] = "type";
    roles[QtRole::ROLE_PARAM_CONTROL] = "control";
    roles[QtRole::ROLE_ISINPUT] = "input";
    roles[QtRole::ROLE_PARAM_GROUP] = "group";
    roles[QtRole::ROLE_PARAM_SOCKET_VISIBLE] = "socket_visible";
    roles[QtRole::ROLE_PARAM_CONTROL_PROPS] = "control_properties";
    roles[QtRole::ROLE_PARAM_SOCKET_CLR] = "socket_color";
    return roles;
}



Q_INVOKABLE bool objParamInputModel::insertRow(int row, QString name)
{
    return false;
}

Q_INVOKABLE bool objParamInputModel::removeRow(int row)
{
    return false;
}

void objParamInputModel::reset()
{
    beginResetModel();
    m_items.clear();
    const zeno::CustomUI& customUI = m_paramsModel->customUI();
    for (const zeno::ParamObject& param : customUI.inputObjs) {
        int r = m_paramsModel->indexFromName(QString::fromStdString(param.name), true);
        QPersistentModelIndex idx = m_paramsModel->index(r);
        m_items.push_back(idx);
    }
    endResetModel();
}

QStandardItemModel* objParamInputModel::toStandardModel() const {
    QStandardItemModel* pModel = new QStandardItemModel;
    for (int j = 0; j < rowCount(); j++) {
        QModelIndex idx = index(j);
        auto paramName = idx.data(QtRole::ROLE_PARAM_NAME).toString();
        QStandardItem* pItem = new QStandardItem(paramName);
        pItem->setData(paramName, QtRole::ROLE_PARAM_NAME);
        pItem->setData(paramName, QtRole::ROLE_PARAM_NAME_EXIST);
        pItem->setData(idx.data(QtRole::ROLE_PARAM_TYPE), QtRole::ROLE_PARAM_TYPE);
        pItem->setData(idx.data(QtRole::ROLE_PARAM_VALUE), QtRole::ROLE_PARAM_VALUE);
        pItem->setData(idx.data(QtRole::ROLE_PARAM_CONTROL), QtRole::ROLE_PARAM_CONTROL);
        pItem->setData(idx.data(QtRole::ROLE_SOCKET_TYPE), QtRole::ROLE_SOCKET_TYPE);
        pItem->setData(idx.data(QtRole::ROLE_ISINPUT), QtRole::ROLE_ISINPUT);
        pItem->setData(idx.data(QtRole::ROLE_NODEIDX), QtRole::ROLE_NODEIDX);
        pItem->setData(idx.data(QtRole::ROLE_PARAM_CONTROL_PROPS), QtRole::ROLE_PARAM_CONTROL_PROPS);
        pItem->setData(idx.data(QtRole::ROLE_PARAM_VISIBLE), QtRole::ROLE_PARAM_VISIBLE);
        pItem->setData(idx.data(QtRole::ROLE_PARAM_SOCKET_VISIBLE), QtRole::ROLE_PARAM_SOCKET_VISIBLE);
        pItem->setData(idx.data(QtRole::ROLE_PARAM_GROUP), QtRole::ROLE_PARAM_GROUP);
        pItem->setData(VPARAM_PARAM, ROLE_ELEMENT_TYPE);
        pModel->appendRow(pItem);
    }
    return pModel;
}

objParamOutputModel::objParamOutputModel(zeno::ObjectParams params, CustomUIModel* pModel)
    : QAbstractListModel(pModel)
    , m_paramsModel(nullptr)
{
    m_paramsModel = pModel->coreModel();
    for (const zeno::ParamObject& param : params) {
        int r = m_paramsModel->indexFromName(QString::fromStdString(param.name), false);
        QPersistentModelIndex idx = m_paramsModel->index(r);
        m_items.push_back(idx);
    }
}

int objParamOutputModel::rowCount(const QModelIndex& parent /*= QModelIndex()*/) const
{
    return m_items.size();
}

QVariant objParamOutputModel::data(const QModelIndex& index, int role /*= Qt::DisplayRole*/) const
{
    return m_items[index.row()].data(role);
}

bool objParamOutputModel::setData(const QModelIndex& index, const QVariant& value, int role /*= Qt::EditRole*/)
{
    return m_paramsModel->setData(m_items[index.row()], value, role);
}

QHash<int, QByteArray> objParamOutputModel::roleNames() const
{
    QHash<int, QByteArray> roles;
    roles[QtRole::ROLE_NODE_NAME] = "nodename";
    roles[QtRole::ROLE_PARAM_NAME] = "paramname";
    roles[QtRole::ROLE_PARAM_TYPE] = "type";
    roles[QtRole::ROLE_PARAM_CONTROL] = "control";
    roles[QtRole::ROLE_ISINPUT] = "input";
    roles[QtRole::ROLE_PARAM_GROUP] = "group";
    roles[QtRole::ROLE_PARAM_SOCKET_VISIBLE] = "socket_visible";
    roles[QtRole::ROLE_PARAM_CONTROL_PROPS] = "control_properties";
    roles[QtRole::ROLE_PARAM_SOCKET_CLR] = "socket_color";
    return roles;
}

Q_INVOKABLE bool objParamOutputModel::insertRow(int row, QString name)
{
    return false;
}


Q_INVOKABLE bool objParamOutputModel::removeRow(int row)
{
    return false;
}

void objParamOutputModel::reset()
{
    beginResetModel();
    m_items.clear();
    const zeno::CustomUI& customUI = m_paramsModel->customUI();
    for (const zeno::ParamObject& param : customUI.outputObjs) {
        int r = m_paramsModel->indexFromName(QString::fromStdString(param.name), false);
        QPersistentModelIndex idx = m_paramsModel->index(r);
        m_items.push_back(idx);
    }
    endResetModel();
}

QStandardItemModel* objParamOutputModel::toStandardModel() const {
    QStandardItemModel* pModel = new QStandardItemModel;
    for (int j = 0; j < rowCount(); j++) {
        QModelIndex idx = index(j);
        auto paramName = idx.data(QtRole::ROLE_PARAM_NAME).toString();
        QStandardItem* pItem = new QStandardItem(paramName);
        pItem->setData(paramName, QtRole::ROLE_PARAM_NAME);
        pItem->setData(paramName, QtRole::ROLE_PARAM_NAME_EXIST);
        pItem->setData(idx.data(QtRole::ROLE_PARAM_TYPE), QtRole::ROLE_PARAM_TYPE);
        pItem->setData(idx.data(QtRole::ROLE_PARAM_VALUE), QtRole::ROLE_PARAM_VALUE);
        pItem->setData(idx.data(QtRole::ROLE_PARAM_CONTROL), QtRole::ROLE_PARAM_CONTROL);
        pItem->setData(idx.data(QtRole::ROLE_SOCKET_TYPE), QtRole::ROLE_SOCKET_TYPE);
        pItem->setData(idx.data(QtRole::ROLE_ISINPUT), QtRole::ROLE_ISINPUT);
        pItem->setData(idx.data(QtRole::ROLE_NODEIDX), QtRole::ROLE_NODEIDX);
        pItem->setData(idx.data(QtRole::ROLE_PARAM_CONTROL_PROPS), QtRole::ROLE_PARAM_CONTROL_PROPS);
        pItem->setData(idx.data(QtRole::ROLE_PARAM_VISIBLE), QtRole::ROLE_PARAM_VISIBLE);
        pItem->setData(idx.data(QtRole::ROLE_PARAM_SOCKET_VISIBLE), QtRole::ROLE_PARAM_SOCKET_VISIBLE);
        pItem->setData(idx.data(QtRole::ROLE_PARAM_GROUP), QtRole::ROLE_PARAM_GROUP);
        pItem->setData(VPARAM_PARAM, ROLE_ELEMENT_TYPE);
        pModel->appendRow(pItem);
    }
    return pModel;
}


ControlItemListModel::ControlItemListModel(CustomUIModel* pModel) : QAbstractListModel(pModel)
{
}

int ControlItemListModel::rowCount(const QModelIndex& parent) const
{
    return sizeof(customui_controlList) / sizeof(CUSTOMUI_CTRL_ITEM_INFO);
}

QVariant ControlItemListModel::data(const QModelIndex& index, int role) const
{
    if (role == QtRole::ROLE_PARAM_NAME) {
        return customui_controlList[index.row()].name;
    } else if (role == QtRole::ROLE_PARAM_CONTROL) {
        return customui_controlList[index.row()].ctrl;
    } else if (role == QtRole::ROLE_PARAM_TYPE) {
        return (quint64)customui_controlList[index.row()].type;
    } else if (role == Qt::DecorationRole) {
        return customui_controlList[index.row()].icon;
    }
    return QVariant();
}

QHash<int, QByteArray> ControlItemListModel::roleNames() const
{
    QHash<int, QByteArray> values;
    values[QtRole::ROLE_PARAM_NAME] = "ctrlItemName";
    values[QtRole::ROLE_PARAM_CONTROL] = "ctrlItemType";
    values[QtRole::ROLE_PARAM_TYPE] = "ctrlContentType";
    values[Qt::DecorationRole] = "ctrlItemICon";
    return values;
}

Qt::ItemFlags ControlItemListModel::flags(const QModelIndex& index) const
{
    if (!index.isValid())
        return Qt::NoItemFlags;
    return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}