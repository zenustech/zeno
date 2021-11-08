#ifndef __ZNODES_GRAPHICSVIEW_H__
#define __ZNODES_GRAPHICSVIEW_H__

#include <QtWidgets>
#include "../qdmgraphicsview.h"
#include "../qdmgraphicsscene.h"

class ZNodesGraphicsView : public QWidget
{
    Q_OBJECT
public:
    ZNodesGraphicsView(QWidget* parent = nullptr);
    void initNodes();

private:
    QDMGraphicsView* m_view;
    QDMGraphicsScene* m_scene;
};

#endif