#ifndef __NODESVIEW_H__
#define __NODESVIEW_H__

class NodeScene;
class NodesView : public QGraphicsView
{
	Q_OBJECT
public:
	NodesView(QWidget* parent = nullptr);
	QSize sizeHint() const override;

protected:
	void mousePressEvent(QMouseEvent* event);

private:
	void initView();

	int m_gridX;
	int m_gridY;
	NodeScene* m_scene;
};


#endif