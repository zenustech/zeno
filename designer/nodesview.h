#ifndef __NODESVIEW_H__
#define __NODESVIEW_H__

class NodeScene;
class NodesView : public QGraphicsView
{
	Q_OBJECT
public:
	NodesView(QWidget* parent = nullptr);
	QSize sizeHint() const override;
	void initSkin(const QString& fn);
	void initNode();

protected:
	void mousePressEvent(QMouseEvent* event);

private:
	int m_gridX;
	int m_gridY;
	NodeScene* m_scene;
};


#endif