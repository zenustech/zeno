#include "qdmnodeparamedit.h"
#include <zeno/ztd/functional.h>
#include <QFormLayout>
#include <QLineEdit>

ZENO_NAMESPACE_BEGIN

QDMNodeParamEdit::QDMNodeParamEdit(QWidget *parent)
    : QWidget(parent)
{
}

static QWidget *make_edit_for_input(dop::Input const &input)
{
    return std::visit(ztd::match
    ( [&] (dop::Input_Value const &v) -> QWidget * {
        return new QLineEdit;
    }
    , [&] (dop::Input_Link const &v) -> QWidget * {
        return new QLineEdit;
    }
    ), input);
}

void QDMNodeParamEdit::setCurrentNode(QDMGraphicsNode *node)
{
    currNode = node;

    auto layout = new QFormLayout;
    for (auto const &[name, input]: node->enumerateSockets()) {
        layout->addRow(name, make_edit_for_input(*input));
    }
    setLayout(layout);
}

QDMNodeParamEdit::~QDMNodeParamEdit() = default;

ZENO_NAMESPACE_END
