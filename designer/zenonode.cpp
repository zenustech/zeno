#include "framework.h"
#include "zenonode.h"
#include <rapidjson/document.h>

using namespace rapidjson;

PictureParam loadBackground(const Value& json)
{
	PictureParam param;
	param.x = json["x"].GetInt();
	param.y = json["y"].GetInt();
	param.w = json["w"].GetInt();
	param.h = json["h"].GetInt();
	param.normal = json["normal"].GetString();
	if (json.HasMember("selected"))
		param.selected = json["selected"].GetString();
	return param;
}

TextParam loadText(const Value& json)
{
	TextParam param;
	param.x = json["x"].GetInt();
	param.y = json["y"].GetInt();
	param.font = QFont(json["font-family"].GetString(), json["font-size"].GetInt());
	//todo: fill
	return param;
}

HeaderParam loadZtfFile(const QString& fileName)
{
	QFile file(fileName);
	bool ret = file.open(QIODevice::ReadOnly | QIODevice::Text);
	Q_ASSERT(ret);
	Document d;
	QByteArray arr = file.readAll();
	HeaderParam param;

	if (!d.Parse(arr).HasParseError())
	{
		Value& status = d["header"]["status"];
		param.once = loadBackground(status["ONCE"]);
		param.mute = loadBackground(status["MUTE"]);
		param.view = loadBackground(status["VIEW"]);
		param.prep = loadBackground(status["PREP"]);

		param.collapse = loadBackground(d["header"]["control"]["collapse"]);
		param.genshin = loadBackground(d["header"]["genshin"]);
		param.background = loadBackground(d["header"]["background"]);
		param.nodename = loadText(d["header"]["nodename"]);
	}
	
	return param;
}


ZenoNode::ZenoNode(QGraphicsItem* parent)
	: QGraphicsObject(parent)
	, m_once(nullptr)
	, m_prep(nullptr)
	, m_mute(nullptr)
	, m_view(nullptr)
	, m_genshin(nullptr)
	, m_background(nullptr)
	, m_nodename(nullptr)
{
	setFlags(ItemIsMovable | ItemSendsGeometryChanges | ItemIsSelectable | ItemClipsToShape);
	initComponent("C:\\editor\\uirender\\node-example.json");
}

void ZenoNode::initComponent(const QString& ztffile)
{
	m_param = loadZtfFile(ztffile);

	m_nodename = new QGraphicsTextItem("Node-name", this);
	QTextDocument* doc = m_nodename->document();
	QTextFrame* rootFrame = doc->rootFrame();
	m_nodename->setDefaultTextColor(QColor(204,204,204));

	m_nodename->setFont(m_param.nodename.font);
	m_background = new QGraphicsPixmapItem(QPixmap(m_param.background.normal).scaled(m_param.background.w, m_param.background.h), this);
	m_once = new QGraphicsPixmapItem(QPixmap(m_param.once.normal).scaled(m_param.once.w, m_param.once.h), this);
	m_mute = new QGraphicsPixmapItem(QPixmap(m_param.mute.normal).scaled(m_param.mute.w, m_param.mute.h), this);
	m_view = new QGraphicsPixmapItem(QPixmap(m_param.view.normal).scaled(m_param.view.w, m_param.view.h), this);
	m_genshin = new QGraphicsPixmapItem(QPixmap(m_param.genshin.normal).scaled(m_param.genshin.w, m_param.genshin.h), this);

	m_nodename->setPos(m_param.nodename.x, m_param.nodename.y);
	m_background->setPos(m_param.background.x, m_param.background.y);
	m_once->setPos(m_param.once.x, m_param.once.y);
	m_mute->setPos(m_param.mute.x, m_param.mute.y);
	m_view->setPos(m_param.view.x, m_param.view.y);
	m_genshin->setPos(m_param.genshin.x, m_param.genshin.y);
}

QRectF ZenoNode::boundingRect() const
{
	QRectF wtf = this->childrenBoundingRect();
	return wtf;
}

QPainterPath ZenoNode::shape() const
{
	return QGraphicsObject::shape();
}

void ZenoNode::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
	//painter->fillRect(boundingRect(), QColor(0,0,0));
}