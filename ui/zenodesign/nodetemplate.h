#ifndef __ZENONODE_H__
#define __ZENONODE_H__

#include <render/renderparam.h>
#include "resizableitemimpl.h"
#include "common.h"

class NodeScene;

class NodeTemplate : public QGraphicsObject
{
	Q_OBJECT
public:
    NodeTemplate(NodeScene* pScene, QGraphicsItem* parent = nullptr);
	void initStyleModel(const NodeParam& param);

	virtual QRectF boundingRect() const override;
	virtual QPainterPath shape() const override;
	virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;
    QStandardItemModel *model() const;
    QItemSelectionModel *selectionModel() const;
    NodeParam exportParam();

protected:
	QVariant itemChange(GraphicsItemChange change, const QVariant& value) override;

signals:
    void markDirty();

public slots:
	void onSelectionChanged(const QItemSelection&, const QItemSelection&);
    void onGvItemSelectedChange(QString id, bool selected);
    void onGvItemGeoChanged(QString id, QRectF sceneRect);
    void onItemChanged(QStandardItem *pItem);
    QStandardItem *getItemFromId(const QString &id);
    void onRowsInserted(const QModelIndex& parent, int first, int last);
    void onDataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight,
                       const QVector<int> &roles = QVector<int>());

private:
    QStandardItem *initModelItemFromGvItem(ResizableItemImpl *pItem, const QString& id, const QString &name);
    void setComonentTxtElem(QStandardItem *pParentItem, ResizableItemImpl *pComponentObj, const TextElement &textElem, const QString &showName);
	void setComponentPxElem(QStandardItem *pParentItem, ResizableItemImpl *pComponentObj, const ImageElement &imgElem, const QString &showName);
    void addImageElement(QStandardItem *pParentItem, const ImageElement &imgElem, ResizableItemImpl *pComponentObj, const QString &showName);
    void addTextElement(QStandardItem *pParentItem, const TextElement &textElem, ResizableItemImpl *pComponentObj, const QString &showName);
    void setupBackground(QStandardItem *pParentItem, ResizableItemImpl *pComponentObj, const BackgroundComponent& bg, const QString& showName);
    SocketComponent _exportSocket(QString id);
    BodyParam _exportBodyParam();
    ImageComponent _exportImageComponent(QString id);
    BackgroundComponent _exportBackgroundComponent(QString id);
    HeaderParam _exportHeaderParam();
    ImageElement _exportImageElement(QString id);
    TextElement _exportTextElement(QString id);
    TextComponent _exportNameComponent(QString id);
    StatusComponent _exportStatusComponent(QString id);
    Component _exportControlComponent(QString id);

	std::map<QString, ResizableItemImpl *> m_objs;

	NodeParam m_param;

	QGraphicsPixmapItem* m_once;
	QGraphicsPixmapItem* m_prep;
	QGraphicsPixmapItem* m_mute;
	QGraphicsPixmapItem* m_view;

	QGraphicsPixmapItem* m_genshin;
	QGraphicsPixmapItem* m_background;
	QGraphicsTextItem* m_nodename;

	ResizableItemImpl* m_component_nodename;
	ResizableItemImpl* m_component_status;
	ResizableItemImpl* m_component_control;
	ResizableItemImpl* m_component_display;
	ResizableItemImpl* m_component_header_backboard;

	ResizableItemImpl* m_component_ltsocket;
	ResizableItemImpl* m_component_lbsocket;
	ResizableItemImpl* m_component_rtsocket;
	ResizableItemImpl* m_component_rbsocket;

	ResizableItemImpl* m_component_body_backboard;

	QStandardItemModel *m_model;
    QItemSelectionModel *m_selection;

	NodeScene *m_pScene;

	int m_batchLevel;
};

#endif