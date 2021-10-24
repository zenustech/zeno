#include "qdmgraphicssocketout.h"
#include <QTextDocument>
#include "qdmgraphicsnode.h"

QDMGraphicsSocketOut::QDMGraphicsSocketOut()
{
    label->setPos(-QDMGraphicsNode::WIDTH - SIZE / 2, -SIZE * 0.7f);
    label->setTextWidth(QDMGraphicsNode::WIDTH);
    auto doc = label->document();
    auto opt = doc->defaultTextOption();
    opt.setAlignment(Qt::AlignRight);
    doc->setDefaultTextOption(opt);
}
