#ifndef __ZEDIT_PARAM_LAYOUT_DLG_H__
#define __ZEDIT_PARAM_LAYOUT_DLG_H__

#include <QtWidgets>
#include <zenomodel/include/viewparammodel.h>

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
    ZEditParamLayoutDlg(ViewParamModel* pModel, bool bNodeUI, const QPersistentModelIndex& nodeIdx, QWidget* parent = nullptr);

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
    void onComboItemsEditFinished();
    void onMinEditFinished();
    void onMaxEditFinished();
    void onStepEditFinished();

private:
    ViewParamModel* m_proxyModel;
    ViewParamModel* m_model;

    Ui::EditParamLayoutDlg* m_ui;
    const QPersistentModelIndex m_index;
};



#endif