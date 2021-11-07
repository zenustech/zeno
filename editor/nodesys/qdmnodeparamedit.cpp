#include "qdmnodeparamedit.h"
#include <QFormLayout>
#include <QLineEdit>

ZENO_NAMESPACE_BEGIN

QDMNodeParamEdit::QDMNodeParamEdit(QWidget *parent)
    : QWidget(parent)
{
}

void QDMNodeParamEdit::setCurrentNode(QDMGraphicsNode *node)
{
    currNode = node;

    auto layout = new QFormLayout;
    for (auto const &[name, edit]: node->enumerateSockets()) {
        layout->addRow(name, edit);
    }
    setLayout(layout);
}

QDMNodeParamEdit::~QDMNodeParamEdit() = default;

ZENO_NAMESPACE_END
