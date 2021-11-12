#ifndef __ZENONODE_H__
#define __ZENONODE_H__


struct PictureParam
{
	QString normal;
	QString selected;
	int fitmode;		//0:  1:  2:
	int x, y, w, h;

	PictureParam() : fitmode(0) {}
};

struct TextParam
{
	QFont font;
	QBrush fill;
	int x, y;
};

struct HeaderParam
{
	PictureParam once;
	PictureParam mute;
	PictureParam view;
	PictureParam prep;
	PictureParam collapse;
	PictureParam genshin;
	PictureParam background;
	TextParam nodename;
};


class ZenoNode : public QGraphicsObject
{
	Q_OBJECT
public:
	ZenoNode(QGraphicsItem* parent = nullptr);

	virtual QRectF boundingRect() const override;
	virtual QPainterPath shape() const override;
	virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

private:
	void initComponent(const QString& ztffile);

private:
	HeaderParam m_param;
	QGraphicsPixmapItem* m_once;
	QGraphicsPixmapItem* m_prep;
	QGraphicsPixmapItem* m_mute;
	QGraphicsPixmapItem* m_view;

	QGraphicsPixmapItem* m_genshin;
	QGraphicsPixmapItem* m_background;
	QGraphicsTextItem* m_nodename;
};

#endif