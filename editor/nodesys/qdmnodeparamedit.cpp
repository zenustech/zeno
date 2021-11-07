#include "qdmnodeparamedit.h"
#include <zeno/ztd/functional.h>
#include <zeno/dop/Descriptor.h>
#include <QFormLayout>
#include <QLineEdit>

ZENO_NAMESPACE_BEGIN

QDMNodeParamEdit::QDMNodeParamEdit(QWidget *parent)
    : QWidget(parent)
{
}

static QWidget *make_edit_for_type(std::string const &type)
{
}

void QDMNodeParamEdit::setCurrentNode(QDMGraphicsNode *node)
{
    currNode = node;

    auto layout = new QFormLayout;
    for (auto const &input: node->getDopNode()->desc->inputs) {
        layout->addRow(QString::fromStdString(input.name),
                       make_edit_for_type(input.type));
    }
    setLayout(layout);
}

QDMNodeParamEdit::~QDMNodeParamEdit() = default;

ZENO_NAMESPACE_END
