#include "qdmgraphicssocketout.h"
#include <QTextDocument>
#include "qdmgraphicsnode.h"

ZENO_NAMESPACE_BEGIN

QDMGraphicsSocketOut::QDMGraphicsSocketOut()
{
    label->setPos(-QDMGraphicsNode::WIDTH - SIZE / 2, -SIZE * 0.7f);
    label->setTextWidth(QDMGraphicsNode::WIDTH);
    auto doc = label->document();
    auto opt = doc->defaultTextOption();
    opt.setAlignment(Qt::AlignRight);
    doc->setDefaultTextOption(opt);
}

QPointF QDMGraphicsSocketOut::getLinkedPos() const
{
    return scenePos() + QPointF(SIZE / 2, 0);
}

ZENO_NAMESPACE_END
