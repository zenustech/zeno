#ifndef __NODESVIEW_H__
#define __NODESVIEW_H__

class NodeScene;
class NodesView : public QGraphicsView
{
	Q_OBJECT
public:
	NodesView(QWidget* parent = nullptr);

private:
	NodeScene* m_scene;
};


#endif