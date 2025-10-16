#ifndef __ZEDIT_PARAM_LAYOUT_DLG_H__
#define __ZEDIT_PARAM_LAYOUT_DLG_H__

/*
 the arch of assets/subgraphs is so complicated and not unit, we have to delay it.
 */

#include <QtWidgets>
#include "model/parammodel.h"


namespace Ui
{
    class EditParamLayoutDlg;
}

struct CONTROL_ITEM_INFO
{
    QString name;
    zeno::ParamControl ctrl;
    zeno::ParamType type;
    QString icon;
};

class ParamTreeItemDelegate : public QStyledItemDelegate
{
    Q_OBJECT
public:
    explicit ParamTreeItemDelegate(QStandardItemModel* model, QObject *parent = nullptr);
    ~ParamTreeItemDelegate();

    // editing
    QWidget* createEditor(QWidget* parent,
        const QStyleOptionViewItem& option,
        const QModelIndex& index) const override;

    void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const override;
    void setEditorData(QWidget* editor, const QModelIndex& index) const override;
    void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override;

    std::function<bool(bool, QString)> m_isGlobalUniqueFunc;

private:
    QStandardItemModel* m_model;
};

class outputListItemDelegate : public QStyledItemDelegate
{
    Q_OBJECT
public:
    explicit outputListItemDelegate(QStandardItemModel* model, zeno::NodeDataGroup group, QObject* parent = nullptr);
    ~outputListItemDelegate();

    // editing
    QWidget* createEditor(QWidget* parent,
        const QStyleOptionViewItem& option,
        const QModelIndex& index) const override;

    void setModelData(QWidget* editor, QAbstractItemModel* model, const QModelIndex& index) const override;
    void setEditorData(QWidget* editor, const QModelIndex& index) const override;
    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;

    std::function<bool(bool, QString)> m_isGlobalUniqueFunc;

private:
    QStandardItemModel* m_model;
    zeno::NodeDataGroup m_group;
};

class ZEditParamLayoutDlg : public QDialog
{
    Q_OBJECT
public:
    ZEditParamLayoutDlg(CustomUIModel* pModel, QWidget* parent = nullptr);
    zeno::ParamsUpdateInfo getEdittedUpdateInfo() const;
    zeno::CustomUI getCustomUiInfo() const;

protected:
    bool eventFilter(QObject* obj, QEvent* event) override;

private slots:
    void onBtnAddInputs();
    void onBtnAddOutputs();
    void onBtnAddObjInputs();
    void onBtnAddObjOutputs();
    void onApply();
    void onOk();
    void onCancel();
    void onTreeCurrentChanged(const QModelIndex& current, const QModelIndex& previous);
    void onOutputsListCurrentChanged(const zeno::NodeDataGroup group, const QModelIndex& current, const QModelIndex& previous);
    void onNameEditFinished();      //name lineedit.
    void onLabelEditFinished();
    void onHintEditFinished();
    void onParamTreeDeleted();
    void onOutputsListDeleted();
    void onObjInputsListDeleted();
    void onObjOutputsListDeleted();
    void onControlItemChanged(int);
    void onParamsViewParamDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles);
    void onOutputsViewParamDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles);
    void onSocketTypeChanged(int idx);
    void onObjTypeChanged(int idx);
    void onOutputPrimTypeChanged(int idx);
    void onMinEditFinished();
    void onMaxEditFinished();
    void onStepEditFinished();
    void onComboTableItemsCellChanged(int row, int column);

private:
    void initUI();
    void initIcon(QStandardItem *item);
    QIcon getIcon(const QStandardItem *pItem);
    void proxyModelSetData(const QModelIndex& index, const zeno::reflect::Any& newValue, int role);
    void switchStackProperties(int ctrl, QStandardItem *pItem);
    void initModel(CustomUIModel* pModel);
    void updateSliderInfo();

    QStringList getExistingNames(bool bInput, VPARAM_TYPE type) const;

    std::function<bool(bool, QString)> m_isGlobalUniqueFunc;

    QStandardItemModel * m_paramsLayoutM_inputs;
    QStandardItemModel * m_paramsLayoutM_outputs;
    QStandardItemModel* m_paramsLayoutM_objInputs;
    QStandardItemModel* m_paramsLayoutM_objOutputs;

    Ui::EditParamLayoutDlg* m_ui;
    const QPersistentModelIndex m_nodeIdx;
    static const int rowValueControl = 3;

    zeno::ParamsUpdateInfo m_paramsUpdate;
    zeno::CustomUI m_customUi;
};

#endif