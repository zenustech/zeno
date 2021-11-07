#include "qdmnodeparamedit.h"
#include <QFormLayout>
#include <QLineEdit>

ZENO_NAMESPACE_BEGIN

QDMNodeParamEdit::QDMNodeParamEdit(QWidget *parent)
    : QWidget(parent)
{
    auto layout = new QFormLayout;
    layout->addRow("hello:", new QLineEdit);
    layout->addRow("this is my name:", new QLineEdit);
    setLayout(layout);
}

QDMNodeParamEdit::~QDMNodeParamEdit() = default;

ZENO_NAMESPACE_END
