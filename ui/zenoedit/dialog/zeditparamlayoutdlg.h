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
    QString icon;
};

class ParamTreeItemDelegate : public QStyledItemDelegate
{
    Q_OBJECT
public:
    explicit ParamTreeItemDelegate(ViewParamModel *model, QObject *parent = nullptr);
    ~ParamTreeItemDelegate();

    // editing
    QWidget* createEditor(QWidget* parent,
        const QStyleOptionViewItem& option,
        const QModelIndex& index) const override;

    void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const override;
    void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override;

  private:
    ViewParamModel *m_model;
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
    void initUI();
    void initIcon(QStandardItem *item);
    QIcon getIcon(const QStandardItem *pItem);
    void initDescValueForProxy();
    void applyForItem(QStandardItem* dstItem, QStandardItem* srcItem);
    void proxyModelSetData(const QModelIndex& index, const QVariant& newValue, int role);
    void recordSubInputCommands(bool bSubInput, VParamItem* pItem);
    void switchStackProperties(int ctrl, VParamItem *pItem);
    void addControlGroup(bool bInput, const QString &name, PARAM_CONTROL ctrl);
    void delControlGroup(bool bInput, const QString &name);
    void updateControlGroup(bool bInput, const QString &newName, const QString &oldName, PARAM_CONTROL ctrl, int row);

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

    bool m_bNodeUI;
};



#endif