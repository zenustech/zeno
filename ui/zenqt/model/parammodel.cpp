#include "parammodel.h"
#include "zassert.h"
#include "util/uihelper.h"
#include <zeno/core/data.h>
#include "LinkModel.h"
#include "GraphModel.h"
#include "customuimodel.h"
#include "variantptr.h"
#include "graphsmanager.h"
#include "GraphsTreeModel.h"
#include <zeno/core/NodeImpl.h>
#include <zeno/utils/helper.h>
#include "declmetatype.h"
#include <zeno/extra/SubnetNode.h>
#include "zenoapplication.h"
#include "style/colormanager.h"

using namespace zeno::types;
using namespace zeno::reflect;


class CustomUIProxyModel : public QStandardItemModel
{
    Q_OBJECT
public:
    CustomUIProxyModel(ParamsModel* parent = nullptr) : QStandardItemModel(parent), m_baseM(parent) {
        connect(m_baseM, &ParamsModel::showPrimSocks_changed, this, &CustomUIProxyModel::showPrimSocks_changed);
    }

    Q_PROPERTY(bool showPrimSocks READ getShowPrimSocks WRITE setShowPrimSocks NOTIFY showPrimSocks_changed)
    
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override {
        //涉及到具体参数的自定义role组别的数据，一律到本model去取，避免冗长而麻烦的两端同步数据，让tree uimodel顶多存一个显示的名字和ui布局。
        //然而，如果走的是QStandardItem->data，它就没法走到这里，不过qml代码没法这么操作，所以widget的情况只能手动在两个模型间同步数据了。
        //但考虑到后面推行quick，而quick不能用standarditem，所以这样做受益
        if (!index.isValid()) {
            //TODO: 要分析一下什么情况会是非法索引
            //ZASSERT_EXIT(false);
            return QStandardItemModel::data(index, role);
        }

        if (role > Qt::UserRole) {
            //一律到本parammodel去取，，让tree uimodel顶多存一个显示的名字和ui布局。
            QStandardItem* pItem = this->itemFromIndex(index);
            QString paramName = pItem->text(); //目前都规定Display的就是真正的名字
            QVariant varGroup = pItem->data(QtRole::ROLE_PARAM_GROUP);
            if (!varGroup.isNull() && varGroup.canConvert<int>()) {
                bool bOk = false;
                zeno::NodeDataGroup group = (zeno::NodeDataGroup)varGroup.toInt(&bOk);
                if (bOk) {
                    if (group == zeno::Role_InputObject || group == zeno::Role_InputPrimitive) {
                        QModelIndex idxparam = m_baseM->paramIdx(paramName, true);
                        if (idxparam.isValid()) {
                            return idxparam.data(role);
                        }
                    }
                    else if (group == zeno::Role_OutputObject || group == zeno::Role_OutputPrimitive) {
                        QModelIndex idxparam = m_baseM->paramIdx(paramName, false);
                        if (idxparam.isValid()) {
                            return idxparam.data(role);
                        }
                    }
                }
            }
        }
        return QStandardItemModel::data(index, role);
    }

    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override {
        if (role > Qt::UserRole) {
            QStandardItem* pItem = this->itemFromIndex(index);
            QString paramName = pItem->text(); //目前都规定Display的就是真正的名字
            QVariant varGroup = pItem->data(QtRole::ROLE_PARAM_GROUP);
            if (!varGroup.isNull() && varGroup.canConvert<int>()) {
                bool bOk = false;
                zeno::NodeDataGroup group = (zeno::NodeDataGroup)varGroup.toInt(&bOk);
                if (bOk) {
                    if (group == zeno::Role_InputObject || group == zeno::Role_InputPrimitive) {
                        QModelIndex idxparam = m_baseM->paramIdx(paramName, true);
                        if (idxparam.isValid()) {
                            return m_baseM->setData(idxparam, value, role);
                        }
                    }
                    else if (group == zeno::Role_OutputObject || group == zeno::Role_OutputPrimitive) {
                        QModelIndex idxparam = m_baseM->paramIdx(paramName, false);
                        if (idxparam.isValid()) {
                            return m_baseM->setData(idxparam, value, role);
                        }
                    }
                }
            }
        }
        return QStandardItemModel::setData(index, value, role);
    }


    bool getShowPrimSocks() const {
        return m_baseM->getShowPrimSocks();
    }

    void setShowPrimSocks(bool bShowPrimSocks) {
        //TODO: 通用设置？
    }

signals:
    void showPrimSocks_changed();

private:
    ParamsModel* m_baseM;
};

#include "parammodel.moc"


class ParamModelImpl : public QObject
{
public:
    ParamModelImpl(std::shared_ptr<zeno::NodeImpl> spNode, QObject* parent)
        : QObject(parent)
        , m_wpNode(spNode)
    {}
    std::weak_ptr<zeno::NodeImpl> m_wpNode;
};

ParamsModel::ParamsModel(const zeno::CustomUI& customui)
    : QAbstractListModel(nullptr)
    , m_customUIM(nullptr)
    , m_inObjProxy(nullptr)
    , m_inPrimProxy(nullptr)
    , m_outPrimProxy(nullptr)
    , m_outObjProxy(nullptr)
    , m_wpNode(nullptr)
    , m_bTempModel(true)
    , m_tempUI(customui)
{
    initProxyModels();
    initParamItems(customui);
    initCustomUI(customui);
}

ParamsModel::ParamsModel(zeno::NodeImpl* spNode, QObject* parent)
    : QAbstractListModel(parent)
    , m_wpNode(spNode)
    , m_customUIM(nullptr)
    , m_inObjProxy(nullptr)
    , m_inPrimProxy(nullptr)
    , m_outPrimProxy(nullptr)
    , m_outObjProxy(nullptr)
{
    initProxyModels();
    initParamItems(m_wpNode->export_customui());
    initCustomUI(spNode->export_customui());

    cbUpdateParam = spNode->register_update_param(
        [this](const std::string& name, zeno::reflect::Any old_value, zeno::reflect::Any new_value) {
            for (int i = 0; i < m_items.size(); i++) {
                if (m_items[i].name.toStdString() == name) {
                    if (m_items[i].value != new_value)
                    {
                        QModelIndex idx = createIndex(i, 0);
                        setData(idx, QVariant::fromValue(new_value), QtRole::ROLE_PARAM_VALUE);
                    }
                    break;
                }
            }
            //根据需要更新节点布局
            auto spNode = m_wpNode/*.lock()*/;
            ZASSERT_EXIT(spNode);
            spNode->trigger_update_params(name, false, zeno::params_change_info());
        });

    spNode->register_update_param_socket_type(
        [this](const std::string& name, zeno::SocketType type) {
            updateParamData(QString::fromStdString(name), type, QtRole::ROLE_SOCKET_TYPE);
        });

    spNode->register_update_param_wildcard(
        [this](const std::string& name, bool isWildcard) {
            updateParamData(QString::fromStdString(name), isWildcard, QtRole::ROLE_PARAM_IS_WILDCARD);
        });

    spNode->register_update_param_type(
        [this](const std::string& name, zeno::ParamType type, bool bInput) {
        updateParamData(QString::fromStdString(name), (quint64)type, QtRole::ROLE_PARAM_TYPE, bInput);
        });

    spNode->register_update_param_control(
        [this](const std::string& name, zeno::ParamControl control) {
            updateParamData(QString::fromStdString(name), control, QtRole::ROLE_PARAM_CONTROL);
        });

    spNode->register_update_param_control_prop(
        [this](const std::string& name, zeno::reflect::Any controlProps) {
            updateParamData(QString::fromStdString(name), QVariant::fromValue(controlProps), QtRole::ROLE_PARAM_CTRL_PROPERTIES);
        });

    spNode->register_update_param_socket_visible(
        [this](const std::string& name, bool bSocketVisible, bool bInput) {
            updateParamData(QString::fromStdString(name), bSocketVisible, QtRole::ROLE_PARAM_SOCKET_VISIBLE, bInput);
        });

    spNode->register_update_visable_enable([this](zeno::NodeImpl* pNode, std::set<std::string> adjInputs, std::set<std::string> adjOutputs) {
        //扫一遍，更新一下缓存值
        for (ParamItem& item : m_items) {
            std::string name = item.name.toStdString();
            if (adjInputs.find(name) != adjInputs.end() && item.bInput) {
                bool bExist = false;
                zeno::CommonParam param = pNode->get_input_param(name, &bExist);
                ZASSERT_EXIT(bExist);
                if (param.bEnable != item.bEnable) {
                    updateParamData(item.name, param.bEnable, QtRole::ROLE_PARAM_ENABLE, true);
                }
                if (param.bVisible != item.bVisible) {
                    updateParamData(item.name, param.bVisible, QtRole::ROLE_PARAM_VISIBLE, true);
                }
            }
            if (adjOutputs.find(name) != adjOutputs.end() && !item.bInput) {
                bool bExist = false;
                zeno::CommonParam param = pNode->get_output_param(name, &bExist);
                ZASSERT_EXIT(bExist);
                if (param.bEnable != item.bEnable) {
                    updateParamData(item.name, param.bEnable, QtRole::ROLE_PARAM_ENABLE, false);
                }
                if (param.bVisible != item.bVisible) {
                    updateParamData(item.name, param.bVisible, QtRole::ROLE_PARAM_VISIBLE, false);
                }
            }
        }
        emit enabledVisibleChanged();
    });

    spNode->register_update_param_color(
        [this](const std::string& name, std::string& clr) {
            updateParamData(QString::fromStdString(name), QString::fromStdString(clr), QtRole::ROLE_PARAM_SOCKET_CLR);
        });

    spNode->register_update_layout(
        [this](zeno::params_change_info& changes) {
            updateUiLinksSockets(changes);
        });
}

void ParamsModel::initProxyModels() {
    m_inObjProxy = new ParamFilterModel(zeno::Role_InputObject, this);
    m_inPrimProxy = new ParamFilterModel(zeno::Role_InputPrimitive, this);
    m_outPrimProxy = new ParamFilterModel(zeno::Role_OutputPrimitive, this);
    m_outObjProxy = new ParamFilterModel(zeno::Role_OutputObject, this);

    m_inObjProxy->setSourceModel(this);
    m_inPrimProxy->setSourceModel(this);
    m_outPrimProxy->setSourceModel(this);
    m_outObjProxy->setSourceModel(this);
}

void ParamsModel::initParamItems(const zeno::CustomUI& customui)
{
    //primitive inputs
    if (!customui.inputPrims.empty() && !customui.inputPrims[0].groups.empty()) {
        for (auto& tab : customui.inputPrims) {
            for (auto& group: tab.groups) {
                auto params = group.params;
                for (const zeno::ParamPrimitive& spParam : params) {
                    ParamItem item;
                    item.bInput = true;
                    item.control = spParam.control;
                    if (item.control == zeno::NullControl)
                        item.control = zeno::getDefaultControl(spParam.type);
                    item.optCtrlprops = spParam.ctrlProps;
                    item.name = QString::fromStdString(spParam.name);
                    item.type = spParam.type;
                    item.value = spParam.defl;
                    item.connectProp = zeno::Socket_Primitve;
                    item.bSocketVisible = spParam.bSocketVisible;
                    item.bVisible = spParam.bVisible;
                    item.bEnable = spParam.bEnable;
                    item.group = zeno::Role_InputPrimitive;
                    item.sockProp = spParam.sockProp;
                    m_items.append(item);
                }
            }
        }
    }
    //object inputs
    for (const auto& spParam : customui.inputObjs) {
        ParamItem item;
        item.bInput = true;
        item.name = QString::fromStdString(spParam.name);
        item.type = spParam.type;
        item.connectProp = spParam.socketType;
        item.group = zeno::Role_InputObject;
        item.bVisible = spParam.bVisible;
        item.bEnable = spParam.bEnable;
        m_items.append(item);
    }

    //primitive outputs
    for (const auto& param : customui.outputPrims) {
        ParamItem item;
        item.bInput = false;
        item.control = zeno::NullControl;
        item.name = QString::fromStdString(param.name);
        item.type = param.type;
        item.connectProp = zeno::Socket_Primitve;
        item.group = zeno::Role_OutputPrimitive;
        item.bSocketVisible = param.bSocketVisible;
        item.bVisible = param.bVisible;
        item.bEnable = param.bEnable;
        m_items.append(item);
    }

    //object outputs
    for (const auto& param : customui.outputObjs) {
        ParamItem item;
        item.bInput = false;
        item.name = QString::fromStdString(param.name);
        item.type = param.type;
        item.connectProp = zeno::Socket_Output;
        item.group = zeno::Role_OutputObject;
        item.bVisible = param.bVisible;
        item.bEnable = param.bEnable;
        m_items.append(item);
    }

    //init custom param model.
    //initCustomUI(spNode->get_customui());
}

zeno::CustomUI ParamsModel::customUI() const {
    if (m_bTempModel && !m_wpNode)
        return m_tempUI;
    else
        return m_wpNode->export_customui();
}

void ParamsModel::initCustomUI(const zeno::CustomUI& customui)
{
#if 0
    QStandardItemModel* legacy_customParamsM = constructProxyModel();
    UiHelper::newCustomModel(legacy_customParamsM, customui);

    //m_customParamsM创建后需更新初始值
    legacy_customParamsM->blockSignals(true);
    zeno::scope_exit sp([=] {legacy_customParamsM->blockSignals(false); });

    QStandardItem* pInputsRoot = legacy_customParamsM->item(0);
    for (int i = 0; i < pInputsRoot->rowCount(); i++)
    {
        auto tabItem = pInputsRoot->child(i);
        for (int j = 0; j < tabItem->rowCount(); j++)
        {
            auto groupItem = tabItem->child(j);
            for (int k = 0; k < groupItem->rowCount(); k++)
            {
                auto paramItem = groupItem->child(k);
                const auto& paramName = paramItem->data(QtRole::ROLE_PARAM_NAME).toString();
                int row = indexFromName(paramName, true);
                if (row != -1)
                {
                    paramItem->setData(QVariant::fromValue(m_items[row].value), QtRole::ROLE_PARAM_VALUE);
                    paramItem->setData(m_items[row].bSocketVisible, QtRole::ROLE_PARAM_SOCKET_VISIBLE);
                    paramItem->setData(m_items[row].bVisible, QtRole::ROLE_PARAM_VISIBLE);
                    paramItem->setData(m_items[row].bEnable, QtRole::ROLE_PARAM_ENABLE);
                }
            }
        }
    }
    QStandardItem* pOutputsRoot = legacy_customParamsM->item(1);
    for (int i = 0; i < pOutputsRoot->rowCount(); i++)
    {
        auto paramItem = pOutputsRoot->child(i);
        const auto& paramName = paramItem->data(QtRole::ROLE_PARAM_NAME).toString();
        int row = indexFromName(paramName, false);
        paramItem->setData(m_items[row].bSocketVisible, QtRole::ROLE_PARAM_SOCKET_VISIBLE);
        paramItem->setData(m_items[row].bVisible, QtRole::ROLE_PARAM_VISIBLE);
        paramItem->setData(m_items[row].bEnable, QtRole::ROLE_PARAM_ENABLE);
    }

    //判断customui是否为默认的情况，即只有一个tab一个group
    if (customui.inputPrims.size() == 1 &&
        customui.inputPrims[0].name == "Tab1" &&
        customui.inputPrims[0].groups.size() == 1 &&
        customui.inputPrims[0].groups[0].name == "Group1" &&
        !dynamic_cast<zeno::SubnetNode*>(m_wpNode) &&
        !m_bTempModel) {
        return;
    }
    else {
        m_customUIM = new CustomUIModel(this, this);
        m_customUIM->initCustomuiConnections(legacy_customParamsM);
    }
#endif
	m_customUIM = new CustomUIModel(this, this);
}

QStandardItemModel* ParamsModel::constructProxyModel()
{
    QStandardItemModel* pModel = new CustomUIProxyModel(this);
    connect(pModel, &QStandardItemModel::dataChanged, [=](const QModelIndex& topLeft, const QModelIndex&, const QVector<int>& roles) {
        for (int role : roles)
        {
            //if (role != QtRole::ROLE_PARAM_VALUE)
            //    continue;

            QVariant newValue = topLeft.data(role);
            QString name = topLeft.data(QtRole::ROLE_PARAM_NAME).toString();
            bool input = topLeft.data(QtRole::ROLE_ISINPUT).toBool();
            const QModelIndex& paramIdx = this->paramIdx(name, input);

            //zeno::scope_exit sp([=] {this->blockSignals(false); });
            //this->blockSignals(true);
            if (!paramIdx.isValid()) {
                continue;
            }
            setData(paramIdx, newValue, role);
        }
        });

    connect(this, &ParamsModel::dataChanged, [=](const QModelIndex& topLeft, const QModelIndex&, const QVector<int>& roles) {
        bool bInput = topLeft.data(QtRole::ROLE_ISINPUT).toBool();
        if (!bInput)
            return;

        for (int role : roles)
        {
            //if (role != QtRole::ROLE_PARAM_VALUE)
            //    continue;

            const QString& name = topLeft.data(QtRole::ROLE_PARAM_NAME).toString();
            Qt::MatchFlags flags = Qt::MatchRecursive | Qt::MatchCaseSensitive;
            auto pItems = pModel->findItems(name, flags);
            for (auto pItem : pItems)
            {
                const QVariant& modelVal = topLeft.data(role);
                zeno::scope_exit sp([=] {pModel->blockSignals(false); });
                pModel->blockSignals(true);
                pItem->setData(modelVal, role);
            }
        }
        });
    return pModel;
}

void ParamsModel::updateCustomUiModelIncremental(const zeno::params_change_info& params, const zeno::CustomUI& customui)
{
    //m_customUIM->updateModelIncremental(params, customui);
    if (m_customUIM)
        m_customUIM->reset();
#if 0
    if (m_customParamsM) {
        UiHelper::udpateCustomModelIncremental(m_customParamsM, params, customui);
    }
    else {
        m_customParamsM = constructProxyModel();
        UiHelper::newCustomModel(m_customParamsM, customui);
    }
    //m_customParamsM创建后需更新初始值
    QStandardItem* pInputsRoot = m_customParamsM->item(0);
    for (int i = 0; i < pInputsRoot->rowCount(); i++)
    {
        auto tabItem = pInputsRoot->child(i);
        for (int j = 0; j < tabItem->rowCount(); j++)
        {
            auto groupItem = tabItem->child(j);
            for (int k = 0; k < groupItem->rowCount(); k++)
            {
                auto paramItem = groupItem->child(k);
                int row = indexFromName(paramItem->data(QtRole::ROLE_PARAM_NAME).toString(), true);
                if (row != -1)
                {
                    paramItem->setData((quint64)m_items[row].type, QtRole::ROLE_PARAM_TYPE);
                    paramItem->setData(m_items[row].control, QtRole::ROLE_PARAM_CONTROL);
                    paramItem->setData(QVariant::fromValue(m_items[row].value), QtRole::ROLE_PARAM_VALUE);
                    paramItem->setData(m_items[row].bSocketVisible, QtRole::ROLE_PARAM_SOCKET_VISIBLE);
                }
            }
        }
    }
    QStandardItem* pOutputsRoot = m_customParamsM->item(1);
    for (int i = 0; i < pOutputsRoot->rowCount(); i++)
    {
        auto paramItem = pOutputsRoot->child(i);
        int row = indexFromName(paramItem->data(QtRole::ROLE_PARAM_NAME).toString(), false);
        if (row != -1) {
            paramItem->setData(m_items[row].bSocketVisible, QtRole::ROLE_PARAM_SOCKET_VISIBLE);
        }
    }
    QStandardItem* pObjInputsRoot = m_customParamsM->item(2);
    for (int i = 0; i < pObjInputsRoot->rowCount(); i++)
    {
        auto paramItem = pObjInputsRoot->child(i);
        int row = indexFromName(paramItem->data(QtRole::ROLE_PARAM_NAME).toString(), true);
        if (row != -1) {
            paramItem->setData(m_items[row].connectProp, QtRole::ROLE_SOCKET_TYPE);
            paramItem->setData(m_items[row].bWildcard, QtRole::ROLE_PARAM_IS_WILDCARD);
        }
    }
#endif
}

bool ParamsModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    ParamItem& param = m_items[index.row()];
    switch (role) {
    case QtRole::ROLE_PARAM_NAME:
        param.name = value.toString();
        break;

    case QtRole::ROLE_PARAM_TYPE:
        param.type = (zeno::ParamType)value.toLongLong();
        break;

    case QtRole::ROLE_PARAM_QML_VALUE:
    {
        //在QML初始化的时候，比如onValueChanged时也会直接调用，要考虑是不是做同值过滤
        bool is_prim_var = false;
        if (param.control == zeno::Lineedit && (param.type == gParamType_Int || param.type == gParamType_Float)) {
            is_prim_var = true;
        }
        if ((param.control == zeno::Vec2edit && (param.type == gParamType_Vec2i || param.type == gParamType_Vec2f)) ||
            (param.control == zeno::Vec3edit && (param.type == gParamType_Vec3i || param.type == gParamType_Vec3f)) ||
            (param.control == zeno::Vec4edit && (param.type == gParamType_Vec4i || param.type == gParamType_Vec4f))) {
            //有些vec3f不一定是PrimVar,比如颜色
            is_prim_var = true;
        }
        const zeno::reflect::Any& anyVal = UiHelper::qvarToAnyByType(value, param.type, is_prim_var);
        setData(index, QVariant::fromValue(anyVal), QtRole::ROLE_PARAM_VALUE);
        break;
    }
    case QtRole::ROLE_PARAM_VALUE:
    {
        const zeno::reflect::Any& anyVal = value.value<zeno::reflect::Any>();
        if (anyVal == param.value) {
            return false;
        }
        param.value = anyVal;
        auto spNode = m_wpNode/*.lock()*/;
        if (spNode) {
            spNode->update_param(param.name.toStdString(), anyVal);
            emit dataChanged(index, index, {QtRole::ROLE_PARAM_QML_VALUE});	//同时要发送信号，让QML端可以捕捉到信号变化从而更新
            break;
            //zenoApp->graphsManager()->currentModel()->markDirty(true);
            //return true;        //the dataChanged signal will be emitted by registered callback function.
        }
        return false;
    }

    case QtRole::ROLE_PARAM_CONTROL:
        param.control = (zeno::ParamControl)value.toInt();
        break;
    case QtRole::ROLE_PARAM_CTRL_PROPERTIES:
        param.optCtrlprops = value.value<zeno::reflect::Any>();
        break;
    case QtRole::ROLE_SOCKET_TYPE:
    {
        auto spNode = m_wpNode/*.lock()*/;
        if (spNode) {
            param.connectProp = (zeno::SocketType)value.toInt();
            spNode->update_param_socket_type(param.name.toStdString(), param.connectProp);
            return true;
        }
        return false;
    }
    case QtRole::ROLE_PARAM_IS_WILDCARD:
    {
        auto spNode = m_wpNode/*.lock()*/;
        if (spNode) {
            param.bWildcard = value.toBool();
            return spNode->update_param_wildcard(param.name.toStdString(), param.bWildcard);
        }
        return false;
    }
    case QtRole::ROLE_PARAM_SOCKET_VISIBLE:
    {
        if (param.sockProp == zeno::Socket_Disable || param.group == zeno::Role_InputObject ||
            param.group == zeno::Role_OutputObject) {
            return false;
        }
        auto spNode = m_wpNode/*.lock()*/;
        if (spNode) {
            spNode->update_param_socket_visible(param.name.toStdString(), value.toBool(), param.bInput);
            emit showPrimSocks_changed();
            return true;
        }
        return false;
    }
    case QtRole::ROLE_PARAM_PERSISTENT_INDEX:
    {
        return false;
    }
    case QtRole::ROLE_NODE_DIRTY:
    {
        if (auto spNode = m_wpNode/*.lock()*/) {
            spNode->mark_dirty(value.toBool());
            return true;
        }
    }
    default:
        return false;
    }

    emit dataChanged(index, index, QVector<int>{role});
    zenoApp->graphsManager()->currentModel()->markDirty(true);
    return true;
}

QVariant ParamsModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid())
        return QVariant();

    int r = index.row();
    if (r >= m_items.size()) {
        return QVariant();
    }

    const ParamItem& param = m_items[r];
    switch (role)
    {
    case QtRole::ROLE_PARAM_NAME:       return param.name;
    case QtRole::ROLE_PARAM_TYPE:       return (quint64)param.type;
    case QtRole::ROLE_PARAM_VALUE:      return QVariant::fromValue(param.value);
    case QtRole::ROLE_PARAM_PERSISTENT_INDEX: return QPersistentModelIndex(index);
    case QtRole::ROLE_PARAM_CONTROL:    return param.control;
    case QtRole::ROLE_SOCKET_TYPE:      return param.connectProp;
    case QtRole::ROLE_PARAM_IS_WILDCARD:return param.bWildcard;
    case QtRole::ROLE_ISINPUT:          return param.bInput;
    case QtRole::ROLE_NODEIDX:          return m_nodeIdx;
    case QtRole::ROLE_LINKS:            return QVariant::fromValue(param.links);
    case QtRole::ROLE_PARAM_SOCKPROP:   return param.sockProp;
    case QtRole::ROLE_PARAM_CTRL_PROPERTIES: {
        if (param.optCtrlprops.has_value())
            return QVariant::fromValue(param.optCtrlprops);
        else
            return QVariant();
    }
    case QtRole::ROLE_PARAM_CONTROL_PROPS: {
        QVariantMap map;
        if (param.optCtrlprops.has_value()) {
            auto typeHash = param.optCtrlprops.type().hash_code();
            if (typeHash == zeno::types::gParamType_StringList) {
                const auto& items = zeno::reflect::any_cast<std::vector<std::string>>(param.optCtrlprops);
                QStringList qitems;
                for (const auto& item : items) {
                    qitems.append(QString::fromStdString(item));
                }
                map["combobox_items"] = qitems;
            }
            else if (typeHash == zeno::types::gParamType_IntList) {
                const auto& items = zeno::reflect::any_cast<std::vector<int>>(param.optCtrlprops);
                QVariantList qitems;
                for (int item : items) {
                    qitems.append(item);
                }
                map["slider"] = qitems;
            }
            else if (typeHash == zeno::types::gParamType_FloatList) {
                const auto& items = zeno::reflect::any_cast<std::vector<float>>(param.optCtrlprops);
                QVariantList qitems;
                for (float item : items) {
                    qitems.append(item);
                }
                map["slider"] = qitems;
            }
        }
        return map;
    }
    case QtRole::ROLE_PARAM_INFO: {
        zeno::ParamPrimitive info;
        info.name = param.name.toStdString();
        info.type = param.type;
        info.control = param.control;
        info.ctrlProps = param.optCtrlprops;
        info.defl = param.value;
        info.socketType = param.connectProp;
        for (auto linkidx : param.links) {
            info.links.push_back(linkidx.data(QtRole::ROLE_LINK_INFO).value<zeno::EdgeInfo>());
        }
        return QVariant::fromValue(info);
    }
    case QtRole::ROLE_NODE_IDX:
    {
        return m_nodeIdx;
    }
    case QtRole::ROLE_NODE_NAME:
    {
        return m_nodeIdx.data(QtRole::ROLE_NODE_NAME);
    }
    case QtRole::ROLE_PARAM_VISIBLE:
    {
        return param.bVisible;
    }
    case QtRole::ROLE_PARAM_ENABLE:
    {
        return param.bEnable;
    }
    case QtRole::ROLE_PARAM_SOCKET_VISIBLE:
    {
        if (param.sockProp == zeno::Socket_Disable)
            return false;
        return param.bSocketVisible;
    }
    case QtRole::ROLE_PARAM_SOCKET_CLR:
    {
        QColor color = ZColorManager::getColorByType(param.type);
        //QList<int> lstClr = { color.red(), color.green(), color.blue() };
        return color;
    }
    case QtRole::ROLE_PARAM_GROUP:
    {
        return param.group;
    }
    case QtRole::ROLE_PARAM_QML_VALUE:
    {
        if (!param.value.has_value()) {
            return QVariant();
        }
        const zeno::ParamType paramType = param.value.type().hash_code();
        switch (paramType)
        {
        case gParamType_String:     return QString::fromStdString(zeno::any_cast_to_string(param.value));
        case gParamType_Float:      return any_cast<float>(param.value);
        case gParamType_Int:        return any_cast<int>(param.value);
        case gParamType_Vec2f: {
            const auto& vec = any_cast<zeno::vec2f>(param.value);
            return QVariantList{ vec[0], vec[1] };
        }
        case gParamType_Vec3f: {
            const auto& vec = any_cast<zeno::vec3f>(param.value);
            return QVariantList{ vec[0], vec[1], vec[2]};
        }
        case gParamType_Vec4f: {
            const auto& vec = any_cast<zeno::vec4f>(param.value);
            return QVariantList{ vec[0], vec[1], vec[2], vec[3] };
        }
        case gParamType_Vec2i: {
            const auto& vec = any_cast<zeno::vec2i>(param.value);
            return QVariantList{ vec[0], vec[1] };
        }
        case gParamType_Vec3i: {
            const auto& vec = any_cast<zeno::vec3i>(param.value);
            return QVariantList{ vec[0], vec[1], vec[2] };
        }
        case gParamType_Vec4i: {
            const auto& vec = any_cast<zeno::vec4i>(param.value);
            return QVariantList{ vec[0], vec[1], vec[2], vec[3] };
        }
        case gParamType_Bool:   return any_cast<bool>(param.value);
        case gParamType_VecEdit: {
            const zeno::vecvar& vecvar = any_cast<zeno::vecvar>(param.value);
            QVariantList vec;
            for (const auto& primvar : vecvar) {
                vec.append(UiHelper::primvarToQVariant(primvar));
            }
            return vec;
        }
        case gParamType_PrimVariant:
        {
            const zeno::PrimVar& var = any_cast<zeno::PrimVar>(param.value);
            return UiHelper::primvarToQVariant(var);
        }
        }
    }
    }
    return QVariant();
}

int ParamsModel::indexFromName(const QString& name, bool bInput) const
{
    for (int i = 0; i < m_items.length(); i++) {
        if (m_items[i].name == name && m_items[i].bInput == bInput) {
            return i;
        }
    }
    return -1;
}

QVariant ParamsModel::getIndexList(bool bInput) const
{
    QVariantList varList;
    for (int i = 0; i < m_items.length(); i++) {
        if (m_items[i].bInput == bInput) {
            varList.append(i);
        }
    }
    return varList;
}

GraphModel* ParamsModel::getGraph() const
{
    QObject* nodeitem = parent();
    if (nodeitem) {
        if (GraphModel* pModel = qobject_cast<GraphModel*>(nodeitem->parent())) {
            return pModel;
        }
    }
    return nullptr;
}

PARAMS_INFO ParamsModel::getInputs()
{
    PARAMS_INFO params_inputs;
    for (ParamItem& item : m_items)
    {
        if (item.bInput)
        {
            zeno::ParamPrimitive info;
            info.name = item.name.toStdString();
            info.type = item.type;
            info.control = item.control;
            info.ctrlProps = item.optCtrlprops;
            info.defl = item.value;
            info.socketType = item.connectProp;
            for (auto linkidx : item.links) {
                info.links.push_back(linkidx.data(QtRole::ROLE_LINK_INFO).value<zeno::EdgeInfo>());
            }
            //info.tooltip //std::string tooltip;
            //info.prop   //SocketProperty prop = Socket_Normal;
            params_inputs.insert(item.name, info);
        }
    }
    return params_inputs;
}

PARAMS_INFO ParamsModel::getOutputs()
{
    PARAMS_INFO params_outputs;
    for (ParamItem& item : m_items)
    {
        if (!item.bInput)
        {
            zeno::ParamPrimitive info;
            info.name = item.name.toStdString();
            info.type = item.type;
            info.control = item.control;
            info.ctrlProps = item.optCtrlprops;
            info.defl = item.value;
            info.socketType = item.connectProp;
            for (auto linkidx : item.links) {
                info.links.push_back(linkidx.data(QtRole::ROLE_LINK_INFO).value<zeno::EdgeInfo>());
            }
            //info.tooltip //std::string tooltip;
            //info.prop   //SocketProperty prop = Socket_Normal;
            params_outputs.insert(item.name, info);
        }
    }
    return params_outputs;
}



QHash<int, QByteArray> ParamsModel::roleNames() const
{
    QHash<int, QByteArray> roles;
    roles[QtRole::ROLE_NODE_NAME] = "nodename";
    roles[QtRole::ROLE_PARAM_NAME] = "name";
    roles[QtRole::ROLE_PARAM_TYPE] = "type";
    roles[QtRole::ROLE_PARAM_CONTROL] = "control";
    roles[QtRole::ROLE_PARAM_QML_VALUE] = "value";
    roles[QtRole::ROLE_ISINPUT] = "input";
    roles[QtRole::ROLE_PARAM_GROUP] = "group";
    roles[QtRole::ROLE_PARAM_SOCKET_VISIBLE] = "socket_visible";
    roles[QtRole::ROLE_PARAM_PERSISTENT_INDEX] = "per_index";
    roles[QtRole::ROLE_PARAM_CONTROL_PROPS] = "control_properties";
    roles[QtRole::ROLE_PARAM_SOCKET_CLR] = "socket_color";
    roles[QtRole::ROLE_PARAM_VISIBLE] = "param_visible";
    return roles;
}

int ParamsModel::rowCount(const QModelIndex& parent) const
{
    return m_items.count();
}

void ParamsModel::setNodeIdx(const QModelIndex& nodeIdx)
{
    m_nodeIdx = nodeIdx;
}

QModelIndex ParamsModel::paramIdx(const QString& name, bool bInput) const
{
    for (int r = 0; r < rowCount(); r++)
    {
        QModelIndex idx = index(r, 0);
        if (name == data(idx, QtRole::ROLE_PARAM_NAME).toString() && bInput == data(idx, QtRole::ROLE_ISINPUT).toBool())
            return idx;
    }
    return QModelIndex();
}

void ParamsModel::addLink(const QModelIndex& paramIdx, const QPersistentModelIndex& linkIdx)
{
    m_items[paramIdx.row()].links.append(linkIdx);
}

int ParamsModel::removeLink(const QModelIndex& paramIdx)
{
    QList<QPersistentModelIndex>& links = m_items[paramIdx.row()].links;
    if (links.isEmpty())
        return -1;

    //ZASSERT_EXIT(links.size() == 1);
    int nRow = links[0].row();
    links.clear();
    return nRow;
}

bool ParamsModel::removeSpecificLink(const QModelIndex& paramIdx, const QModelIndex& linkIdx)
{
    int row = paramIdx.row();
    if (row < 0 || row >= m_items.size())
        return false;

    QList<QPersistentModelIndex>& links = m_items[row].links;
    for (auto link : links) {
        if (link == linkIdx) {
            links.removeOne(link);
            return true;
        }
    }
    return false;
}

QModelIndex ParamsModel::removeOneLink(const QModelIndex& paramIdx, const zeno::EdgeInfo& link)
{
    QList<QPersistentModelIndex>& links = m_items[paramIdx.row()].links;
    if (links.isEmpty())
        return QModelIndex();

    for (auto it = links.begin(); it != links.end(); it++) {
        const zeno::EdgeInfo& lnk = (*it).data(QtRole::ROLE_LINK_INFO).value<zeno::EdgeInfo>();
        if (lnk == link) {
            QModelIndex idx = *it;
            it = links.erase(it);
            return idx;
        }
    }
    return QModelIndex();
}

void ParamsModel::addParam(const ParamItem& param)
{
    int nRows = m_items.size();
    beginInsertRows(QModelIndex(), nRows, nRows);
    m_items.append(param);

    if (m_wpNode) {
        if (param.group == zeno::Role_InputPrimitive || param.group == zeno::Role_OutputPrimitive) {
            zeno::ParamPrimitive info;
            info.bInput = param.bInput;
            info.name = param.name.toStdString();
            info.type = param.type;
            info.control = param.control;
            info.ctrlProps = param.optCtrlprops;
            info.defl = param.value;
            info.socketType = param.connectProp;

            info.bSocketVisible = param.bSocketVisible;
            info.bVisible = param.bVisible;
            info.bEnable = param.bEnable;
            info.bWildcard = param.bWildcard;
            if (param.bInput) {
                m_wpNode->add_input_prim_param(info);
            } else {
                m_wpNode->add_output_prim_param(info);
            }
        } else {
            zeno::ParamObject info;
            info.bInput = param.bInput;
            info.type = param.type;
            info.name = param.name.toStdString();
            info.socketType = param.connectProp;
            if (param.bInput) {
                m_wpNode->add_input_obj_param(info);
            }
            else {
                m_wpNode->add_output_obj_param(info);
            }
        }
    }
    endInsertRows();
}

void ParamsModel::removeParam(const QString& name, bool bInput)
{
    for (int i = 0; i < m_items.size(); i++) {
        if (m_items[i].name == name && m_items[i].bInput == bInput) {
            beginRemoveRows(QModelIndex(), i, i);
            m_items.erase(m_items.begin() + i);
            endRemoveRows();
            return;
        }
    }
}

GraphModel* ParamsModel::parentGraph() const
{
    QObject* nodeitem = parent();
    if (nodeitem) {
        if (GraphModel* pModel = qobject_cast<GraphModel*>(nodeitem->parent())) {
            return pModel;
        }
    }
    return nullptr;
}

QStandardItemModel* ParamsModel::customParamModel()
{
    return nullptr;
    //return m_customParamsM;
}

CustomUIModel* ParamsModel::customUIModel()
{
    return m_customUIM;
}

ParamFilterModel* ParamsModel::inputObjects() {
    return m_inObjProxy;
}

ParamFilterModel* ParamsModel::inputPrims() {
    return m_inPrimProxy;
}

ParamFilterModel* ParamsModel::outputPrims() {
    return m_outPrimProxy;
}

ParamFilterModel* ParamsModel::outputObjects() {
    return m_outObjProxy;
}

Qt::ItemFlags ParamsModel::flags(const QModelIndex& index) const
{
    if (!index.isValid())
        return Qt::NoItemFlags;
    return Qt::ItemIsEditable | QAbstractListModel::flags(index);
}

void ParamsModel::batchModifyParams(const zeno::ParamsUpdateInfo& params, bool bSubnetInit)
{
    //if (params.empty())   //可能是删除到空的情况，无需return
    //    return;

    auto spNode = m_wpNode/*.lock()*/;
    ZASSERT_EXIT(spNode);
    this->blockSignals(this);   //updateParamData不发出的datachange信号否则触发m_customParamsM的datachange
    zeno::params_change_info changes = spNode->update_editparams(params, bSubnetInit);
    this->blockSignals(false);
    updateUiLinksSockets(changes);
}

void ParamsModel::updateUiLinksSockets(zeno::params_change_info& changes)
{
    auto spNode = m_wpNode/*.lock()*/;
    ZASSERT_EXIT(spNode);

    //assuming that the param layout has changed, and we muse reconstruct all params and index.
    emit layoutAboutToBeChanged();

    //remove old links from this node.
    for (int r = 0; r < m_items.size(); r++) {
        ParamItem& item = m_items[r];
        for (QPersistentModelIndex linkIdx : item.links) {
            if (item.bInput) {
                QModelIndex outSockIdx = linkIdx.data(QtRole::ROLE_OUTSOCK_IDX).toModelIndex();
                //only remove link by model itself, with no action about core data.
                QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(outSockIdx.model());
                ParamsModel* outParams = qobject_cast<ParamsModel*>(pModel);
                ZASSERT_EXIT(outParams);
                bool ret = outParams->removeSpecificLink(outSockIdx, linkIdx);
                ZASSERT_EXIT(ret);
            }
            else {
                QModelIndex inSockIdx = linkIdx.data(QtRole::ROLE_INSOCK_IDX).toModelIndex();
                //only remove link by model itself, with no action about core data.
                QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(inSockIdx.model());
                ParamsModel* inParams = qobject_cast<ParamsModel*>(pModel);
                ZASSERT_EXIT(inParams);
                bool ret = inParams->removeSpecificLink(inSockIdx, linkIdx);
                ZASSERT_EXIT(ret);
            }
        }

        for (QPersistentModelIndex linkIdx : item.links) {
            QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(linkIdx.model());
            LinkModel* linkModel = qobject_cast<LinkModel*>(pModel);
            ZASSERT_EXIT(linkModel);
            //no standard api to user, just delete from model, and sync to ui.
            linkModel->removeRows(linkIdx.row(), 1);
        }

        item.links.clear();
    }

    m_items.clear();
    //reconstruct params.
    initParamItems(m_wpNode->export_customui());
    //TODO: 尽量废弃dynamic_cast这种写法
    if (auto sbn = dynamic_cast<zeno::SubnetNode*>(spNode)) {
        updateCustomUiModelIncremental(changes, sbn->get_customui());
    }
    else {
        updateCustomUiModelIncremental(changes, spNode->export_customui());
    }

    //reconstruct links.
    for (int r = 0; r < m_items.size(); r++) {
        auto group = m_items[r].group;
        std::vector<zeno::EdgeInfo> links;
        if (group == zeno::Role_InputPrimitive)
        {
            bool bExist = false;
            auto paramPrim = spNode->get_input_prim_param(m_items[r].name.toStdString(), &bExist);
            if (!bExist)
                continue;
            links = paramPrim.links;

        }
        else if (group == zeno::Role_InputObject)
        {
            bool bExist = false;
            auto paramObj = spNode->get_input_obj_param(m_items[r].name.toStdString(), &bExist);
            if (!bExist)
                continue;
            links = paramObj.links;
        }
        else if (group == zeno::Role_OutputPrimitive)
        {
            bool bExist = false;
            auto paramPrim = spNode->get_output_prim_param(m_items[r].name.toStdString(), &bExist);
            if (!bExist)
                continue;
            links = paramPrim.links;
        }
        else if (group == zeno::Role_OutputObject)
        {
            bool bExist = false;
            auto paramPrim = spNode->get_output_obj_param(m_items[r].name.toStdString(), &bExist);
            if (!bExist)
                continue;
            links = paramPrim.links;
        }
        for (const auto& linkInfo : links) {

            const QString fromNode = QString::fromStdString(linkInfo.outNode);
            const QString toNode = QString::fromStdString(linkInfo.inNode);
            const QString fromSock = QString::fromStdString(linkInfo.outParam);
            const QString toSock = QString::fromStdString(linkInfo.inParam);
            const QString outKey = QString::fromStdString(linkInfo.outKey);
            const QString inKey = QString::fromStdString(linkInfo.inKey);

            //add the new link in current graph.
            GraphModel* pGraphM = parentGraph();
            QModelIndex fromNodeIdx = pGraphM->indexFromName(fromNode);
            QModelIndex toNodeIdx = pGraphM->indexFromName(toNode);
            ZASSERT_EXIT(fromNodeIdx.isValid() && toNodeIdx.isValid());

            ParamsModel* fromParams = QVariantPtr<ParamsModel>::asPtr(fromNodeIdx.data(QtRole::ROLE_PARAMS));
            ParamsModel* toParams = QVariantPtr<ParamsModel>::asPtr(toNodeIdx.data(QtRole::ROLE_PARAMS));
            ZASSERT_EXIT(fromParams && toParams);
            QModelIndex fromParam = fromParams->paramIdx(fromSock, false);
            QModelIndex toParam = toParams->paramIdx(toSock, true);
            ZASSERT_EXIT(fromParam.isValid() && toParam.isValid());

            LinkModel* lnkModel = pGraphM->getLinkModel();
            ZASSERT_EXIT(lnkModel);
            //only add in model layer, not core layer.
            QModelIndex newLink = lnkModel->addLink(fromParam, outKey, toParam, inKey, linkInfo.bObjLink);

            fromParams->m_items[fromParam.row()].links.append(newLink);
            toParams->m_items[toParam.row()].links.append(newLink);
        }
    }
    //resetCustomParamModel();

    initProxyModels();
    emit layoutChanged();
}

void ParamsModel::updateParamData(const QString& name, const QVariant& val, int role, bool bInput)
{
    for (int i = 0; i < m_items.size(); i++) {
        if (m_items[i].name == name && m_items[i].bInput == bInput) {
            if (role == QtRole::ROLE_PARAM_CONTROL)
                m_items[i].control = (zeno::ParamControl)val.toInt();
            else if (role == QtRole::ROLE_PARAM_TYPE)
                m_items[i].type = val.value<zeno::ParamType>();
            else if (role == QtRole::ROLE_SOCKET_TYPE)
                m_items[i].connectProp = (zeno::SocketType)val.toInt();
            else if (role == QtRole::ROLE_PARAM_CTRL_PROPERTIES)
                m_items[i].optCtrlprops = val.value<zeno::reflect::Any>();
            else if (role == QtRole::ROLE_PARAM_IS_WILDCARD)
                m_items[i].bWildcard = val.toBool();
            else if (role == QtRole::ROLE_PARAM_SOCKET_VISIBLE) {
                if (m_items[i].bInput == bInput)
                    m_items[i].bSocketVisible = val.toBool();
                else
                    continue;
            }
            else if (role == QtRole::ROLE_PARAM_ENABLE) {
                m_items[i].bEnable = val.toBool();
            }
            else if (role == QtRole::ROLE_PARAM_VISIBLE) {
                m_items[i].bVisible = val.toBool();
            }
            else if (role == QtRole::ROLE_PARAM_GROUP)
                m_items[i].group = (zeno::NodeDataGroup)val.toInt();
            else if (role == QtRole::ROLE_PARAM_SOCKET_CLR) {
            }
            QModelIndex idx = createIndex(i, 0);
            emit dataChanged(idx, idx, { role });
            break;
        }
    }
}

void ParamsModel::resetCustomUi(const zeno::CustomUI& customui)
{
    auto spNode = m_wpNode/*.lock()*/;
    if (auto sbn = dynamic_cast<zeno::SubnetNode*>(spNode))
        sbn->setCustomUi(customui);
}

bool ParamsModel::removeRows(int row, int count, const QModelIndex& parent)
{
    beginRemoveRows(parent, row, row);
    m_items.removeAt(row);
    endRemoveRows();
    return true;
}

void ParamsModel::getDegrees(int& inDegrees, int& outDegrees) {
    inDegrees = outDegrees = 0;
    for (auto item : m_items) {
        if (item.bInput) {
            inDegrees += item.links.size();
        }
        else {
            outDegrees += item.links.size();
        }
    }
}

bool ParamsModel::hasVisiblePrimParam() const {
    for (auto item : m_items) {
        if (item.bSocketVisible &&
            (item.group == zeno::Role_InputPrimitive || item.group == zeno::Role_OutputPrimitive))
        {
            return true;
        }
    }
    return false;
}

int ParamsModel::getParamlinkCount(const QModelIndex& paramIdx)
{
    return m_items[paramIdx.row()].links.size();
}

int ParamsModel::numOfInputParams() const
{
    int n = 0;
    for (auto item : m_items) {
        if (item.bInput)
            n++;
    }
    return n;
}

QString ParamsModel::getMaxLengthName() const {
    QString maxName;
    for (auto& item : m_items) {
        if (item.group == zeno::Role_InputPrimitive) {
            if (item.name.length() > maxName.length()) {
                maxName = item.name;
            }
        }
    }
    return maxName;
}

int ParamsModel::getNumOfOutputPrims() const {
    int n = 0;
    for (auto item : m_items) {
        if (item.group == zeno::Role_OutputPrimitive)
            n++;
    }
    return n;
}

bool ParamsModel::getShowPrimSocks() const {
    for (auto& item : m_items) {
        if (item.group == zeno::Role_InputPrimitive || item.group == zeno::Role_OutputPrimitive) {
            if (item.bSocketVisible)
                return true;
        }
    }
    return false;
}

ParamFilterModel::ParamFilterModel(zeno::NodeDataGroup group, QObject* parent)
    : QSortFilterProxyModel(parent)
    , m_group(group)
{
}

int ParamFilterModel::indexFromName(const QString& target_name) const {
    for (int i = 0; i < rowCount(); i++) {
        QModelIndex idx = index(i, 0);
        QString name = idx.data(QtRole::ROLE_PARAM_NAME).toString();
        if (target_name == name) {
            return i;
        }
    }
    return -1;
}

bool ParamFilterModel::filterAcceptsRow(int sourceRow, const QModelIndex& sourceParent) const {
    QModelIndex srcIdx = sourceModel()->index(sourceRow, 0, sourceParent);
    return srcIdx.data(QtRole::ROLE_PARAM_GROUP) == m_group;
}
