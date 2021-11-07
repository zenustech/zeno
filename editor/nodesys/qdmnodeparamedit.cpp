#include "qdmnodeparamedit.h"
#include <zeno/ztd/functional.h>
#include <zeno/ztd/algorithm.h>
#include <zeno/dop/Descriptor.h>
#include <QLineEdit>
#include <QIntValidator>
#include <QDoubleValidator>

ZENO_NAMESPACE_BEGIN

QDMNodeParamEdit::QDMNodeParamEdit(QWidget *parent)
    : QWidget(parent)
    , layout(new QFormLayout)
{
}

static QWidget *make_edit_for_type(std::string const &type)
{
    static const std::array tab = {
        "string",
        "int",
        "float",
    };
    switch (ztd::try_find_index(tab, type)) {
    case 0: {
        auto edit = new QLineEdit;
        return edit;
    } break;
    case 1: {
        auto edit = new QLineEdit;
        edit->setValidator(new QIntValidator);
        return edit;
    } break;
    case 2: {
        auto edit = new QLineEdit;
        edit->setValidator(new QDoubleValidator);
        return edit;
    } break;
    default: {
        auto edit = new QLineEdit;
        return edit;
    } break;
    }
}

void QDMNodeParamEdit::setCurrentNode(QDMGraphicsNode *node)
{
    while (layout->rowCount())
        layout->removeRow(0);

    currNode = node;
    if (!node)
        return;

    for (auto const &input: node->getDopNode()->desc->inputs) {
        layout->addRow(QString::fromStdString(input.name),
                       make_edit_for_type(input.type));
    }
    setLayout(layout);
}

QDMNodeParamEdit::~QDMNodeParamEdit() = default;

ZENO_NAMESPACE_END
