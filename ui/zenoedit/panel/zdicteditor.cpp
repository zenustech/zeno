#include "zdicteditor.h"
#include "ui_zdicteditor.h"
#include <zenoui/model/dictmodel.h>


ZDictEditor::ZDictEditor(DictModel* pModel, QWidget* parent)
    : QWidget(parent)
{
    m_ui = new Ui::DictEditor;
    m_ui->setupUi(this);

    //delegate model
    m_model = new QStandardItemModel;
    m_model->setColumnCount(3);
    m_model->setHeaderData(0, Qt::Horizontal, tr("Key"));
    m_model->setHeaderData(1, Qt::Horizontal, tr("Type"));
    m_model->setHeaderData(2, Qt::Horizontal, tr("Value"));

    for (int r = 0; r < pModel->rowCount(); r++)
    {
        const QModelIndex& idx = pModel->index(r, 0);
        const QString& key = idx.data(ROLE_KEY).toString();
        const QString& type = idx.data(ROLE_DATATYPE).toString();
        const QVariant& var = idx.data(ROLE_VALUE);

        QStandardItem *pKeyItem = new QStandardItem(key);
        QStandardItem *pTypeItem = new QStandardItem(type);
        QStandardItem *pValueItem = new QStandardItem(var.toInt());

        m_model->appendRow({pKeyItem, pTypeItem, pValueItem});
    }

    m_ui->tableView->setModel(m_model);

    connect(m_ui->btnAdd, SIGNAL(clicked()), this, SLOT(onAddClicked()));
}

void ZDictEditor::onAddClicked()
{
    QStandardItem *pKeyItem = new QStandardItem(tr("please input key"));
    QStandardItem *pTypeItem = new QStandardItem("");
    QStandardItem *pValueItem = new QStandardItem();
    m_model->appendRow({pKeyItem, pTypeItem, pValueItem});
}

