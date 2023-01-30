#ifndef __ZEDIT_PARAM_LAYOUT_DLG_H__
#define __ZEDIT_PARAM_LAYOUT_DLG_H__

#include <QtWidgets>
#include <zenomodel/include/viewparammodel.h>
#include <zenomodel/include/vparamitem.h>

namespace Ui
{
    class EditParamLayoutDlg;
}

struct CONTROL_ITEM_INFO
{
    QString name;
    PARAM_CONTROL ctrl;
    QString defaultType;
};

class ParamTreeItemDelegate : public QStyledItemDelegate
{
    Q_OBJECT
public:
    explicit ParamTreeItemDelegate(QObject* parent = nullptr);
    ~ParamTreeItemDelegate();

    // editing
    QWidget* createEditor(QWidget* parent,
        const QStyleOptionViewItem& option,
        const QModelIndex& index) const override;
};


class ZEditParamLayoutDlg : public QDialog
{
    Q_OBJECT
public:
    ZEditParamLayoutDlg(QStandardItemModel* pModel, bool bNodeUI, const QPersistentModelIndex& nodeIdx, IGraphsModel* pGraphsModel, QWidget* parent = nullptr);

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
    void onTypeItemChanged(int);
    void onComboTableItemsCellChanged(int row, int column);
    void onProxyItemNameChanged(const QModelIndex& itemIdx, const QString& oldPath, const QString& newName);

private:
    void applyForItem(QStandardItem* dstItem, QStandardItem* srcItem);
    void proxyModelSetData(const QModelIndex& index, const QVariant& newValue, int role);
    void recordSubInputCommands(bool bSubInput, VParamItem* pItem);

    ViewParamModel* m_proxyModel;
    ViewParamModel* m_model;
    IGraphsModel* m_pGraphsModel;

    Ui::EditParamLayoutDlg* m_ui;
    const QPersistentModelIndex m_nodeIdx;
    QPersistentModelIndex m_subgIdx;
    static const int rowValueControl = 4;
    bool m_bSubgraphNode;

    QMap<QString, QString> m_renameRecord;
    QVector<QUndoCommand*> m_commandSeq;
};



#endif