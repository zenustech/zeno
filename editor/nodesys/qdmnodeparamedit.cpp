#include "qdmnodeparamedit.h"
#include <zeno/dop/Descriptor.h>

ZENO_NAMESPACE_BEGIN

QDMNodeParamEdit::QDMNodeParamEdit(QWidget *parent)
    : QListView(parent)
    , model(new QStandardItemModel(this))
{
    for (auto const &[k, d]: dop::descriptor_table()) {
        auto item = new QStandardItem();
        item->setText(QString::fromStdString(k));
        item->setEditable(false);
        items.emplace_back(item);
        model->appendRow(item);
    }

    setModel(model.get());
}

QDMNodeParamEdit::~QDMNodeParamEdit() = default;

ZENO_NAMESPACE_END
