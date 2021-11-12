#ifndef __NODES_SCENE_H__
#define __NODES_SCENE_H__



class NodeScene : public QGraphicsScene
{
	Q_OBJECT
public:
	NodeScene(QObject* parent = nullptr);

private:
	void initNode();
};


#endif