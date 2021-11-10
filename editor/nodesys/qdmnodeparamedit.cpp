#include "qdmnodeparamedit.h"
#include <zeno/ztd/variant.h>
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

static const std::array edit_type_table = {
    "string",
    "int",
    "float",
};

QWidget *QDMNodeParamEdit::make_edit_for_type(
    QDMGraphicsNode *node, int sockid, std::string const &type)
{
    auto *input = node->socketInAt(sockid)->value;

    switch (ztd::try_find_index(edit_type_table, type)) {

    case 0: {
        auto edit = new QLineEdit;

        if (auto expr = input->value_cast<std::string>()) {
            auto const &value = *expr;
            edit->setText(QString::fromStdString(value));
        }

        connect(edit, &QLineEdit::editingFinished, this, [=, this] {
            auto expr = edit->text().toStdString();
            auto const &value = expr;
            *input = ztd::make_any(value);
            emit nodeParamUpdated(node);
        });
        return edit;
    } break;

    case 1: {
        auto edit = new QLineEdit;
        edit->setValidator(new QIntValidator);

        if (auto expr = input->value_cast<int>()) {
            auto value = std::to_string(*expr);
            edit->setText(QString::fromStdString(value));
        }

        connect(edit, &QLineEdit::editingFinished, this, [=, this] {
            auto expr = edit->text().toStdString();
            auto value = std::stoi(expr);
            *input = ztd::make_any(value);
            emit nodeParamUpdated(node);
        });
        return edit;
    } break;

    case 2: {
        auto edit = new QLineEdit;
        edit->setValidator(new QDoubleValidator);

        if (auto expr = input->value_cast<float>()) {
            auto value = std::to_string(*expr);
            edit->setText(QString::fromStdString(value));
        }

        connect(edit, &QLineEdit::editingFinished, this, [=, this] {
            auto expr = edit->text().toStdString();
            auto value = std::stof(expr);
            *input = ztd::make_any(value);
            emit nodeParamUpdated(node);
        });
        return edit;
    } break;

    default: {
        return nullptr;
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

    auto dopNode = node->getDopNode();
    for (size_t i = 0; i < dopNode->inputs.size(); i++) {
        auto const &input = dopNode->desc->inputs.at(i);
        auto edit = make_edit_for_type(node, input.type, &dopNode->inputs.at(i));
        if (edit) {
            layout->addRow(QString::fromStdString(input.name), edit);
        }
    }
}

QDMNodeParamEdit::~QDMNodeParamEdit() = default;

ZENO_NAMESPACE_END
