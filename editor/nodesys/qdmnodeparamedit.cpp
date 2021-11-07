#include "qdmnodeparamedit.h"
#include <zeno/ztd/functional.h>
#include <zeno/dop/Descriptor.h>
#include <QFormLayout>
#include <QLineEdit>
#include <QIntValidator>
#include <QDoubleValidator>

ZENO_NAMESPACE_BEGIN

QDMNodeParamEdit::QDMNodeParamEdit(QWidget *parent)
    : QWidget(parent)
{
}

static QWidget *make_edit_for_type(std::string const &type)
{
    const std::array tab = {
        "string",
        "int",
        "float",
    };
    switch (std::find(begin(tab), end(tab), type) - begin(tab)) {
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
