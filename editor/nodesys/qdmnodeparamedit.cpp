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
    , layout(new QFormLayout(this))
{
}

QWidget *QDMNodeParamEdit::make_edit_for_type(std::string const &type, std::function<void(QString)> const &func)
{
    static const std::array tab = {
        "string",
        "int",
        "float",
    };
    switch (ztd::try_find_index(tab, type)) {
    case 0: {
        auto edit = new QLineEdit;
        connect(edit, &QLineEdit::editingFinished, this, [=, this] {
            func(edit->text());
        });
        return edit;
    } break;
    case 1: {
        auto edit = new QLineEdit;
        edit->setValidator(new QIntValidator);
        connect(edit, &QLineEdit::editingFinished, this, [=, this] {
            func(edit->text());
        });
        return edit;
    } break;
    case 2: {
        auto edit = new QLineEdit;
        edit->setValidator(new QDoubleValidator);
        connect(edit, &QLineEdit::editingFinished, this, [=, this] {
            func(edit->text());
        });
        return edit;
    } break;
    default: {
        auto edit = new QLineEdit;
        connect(edit, &QLineEdit::editingFinished, this, [=, this] {
            func(edit->text());
        });
        return edit;
    } break;
    }
}

static void set_input_from_text(dop::Input &input, std::string const &type, std::string const &text) {
}

void QDMNodeParamEdit::setCurrentNode(QDMGraphicsNode *node)
{
    while (layout->rowCount())
        layout->removeRow(0);

    currNode = node;
    if (!node)
        return;

    auto dopNode = node->getDopNode();
    for (size_t i = 0; i < dopNode->inputs.size(); i++) {
        auto const &input = dopNode->desc->inputs.at(i);
        auto type = input.type;
        layout->addRow(QString::fromStdString(input.name),
                       make_edit_for_type(type, [=] (QString text) {
                           set_input_from_text(dopNode->inputs.at(i), type, text.toStdString());
                       }));
    }
}

QDMNodeParamEdit::~QDMNodeParamEdit() = default;

ZENO_NAMESPACE_END
