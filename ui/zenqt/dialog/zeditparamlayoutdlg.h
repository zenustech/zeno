#ifndef __ZEDIT_PARAM_LAYOUT_DLG_H__
#define __ZEDIT_PARAM_LAYOUT_DLG_H__

/*
 the arch of assets/subgraphs is so complicated and not unit, we have to delay it.
 */

#include <QtWidgets>
#include "model/parammodel.h"
#include <zeno/core/data.h>


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
    explicit ParamTreeItemDelegate(ParamsModel* model, QObject *parent = nullptr);
    ~ParamTreeItemDelegate();

    // editing
    QWidget* createEditor(QWidget* parent,
        const QStyleOptionViewItem& option,
        const QModelIndex& index) const override;

    void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const override;
    void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override;

private:
    ParamsModel* m_model;
};


class ZEditParamLayoutDlg : public QDialog
{
    Q_OBJECT
public:
    ZEditParamLayoutDlg(ParamsModel* pModel, QWidget* parent = nullptr);

private slots:
    void onBtnAdd();
    void onApply();
    void onOk();
    void onCancel();
    void onTreeCurrentChanged(const QModelIndex& current, const QModelIndex& previous);
    void onNameEditFinished();      //name lineedit.
    void onLabelEditFinished();
    void onHintEditFinished();
    void onParamTreeDeleted();
    void onChooseParamClicked();
    void onMinEditFinished();
    void onMaxEditFinished();
    void onStepEditFinished();
    void onControlItemChanged(int);
    void onComboTableItemsCellChanged(int row, int column);
    void onProxyItemNameChanged(const QModelIndex& itemIdx, const QString& oldPath, const QString& newName);
    void onViewParamDataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight, const QVector<int> &roles);

private:
    void initUI();
    void _initLayoutModel();
    void initIcon(QStandardItem *item);
    QIcon getIcon(const QStandardItem *pItem);
    void proxyModelSetData(const QModelIndex& index, const QVariant& newValue, int role);
    void switchStackProperties(int ctrl, QStandardItem *pItem);
    void updateSliderInfo();

    ParamsModel* m_model;
    QStandardItemModel* m_paramsLayoutM;

    Ui::EditParamLayoutDlg* m_ui;
    const QPersistentModelIndex m_nodeIdx;
    static const int rowValueControl = 3;
    bool m_bSubgraphNode;

    QMap<QString, QString> m_renameRecord;
    QVector<QUndoCommand*> m_commandSeq;

    bool m_bNodeUI;
};

#endif