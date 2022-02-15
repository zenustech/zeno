#ifndef __ZENO_NEWMENU_H__
#define __ZENO_NEWMENU_H__

#include <QtWidgets>
#include <zenoui/model/modeldata.h>

class ZenoNewnodeMenu : public QMenu
{
	Q_OBJECT
public:
	ZenoNewnodeMenu(const QModelIndex& subgIdx, const NODE_CATES& cates, const QPointF& scenePos, QWidget* parent = nullptr);
	~ZenoNewnodeMenu();

public slots:
	void onTextChanged(const QString& text);

private:
	const NODE_CATES m_cates;
	const QModelIndex m_subgIdx;
	const QPointF m_scenePos;
	QLineEdit* m_searchEdit;
	QWidgetAction* m_pWAction;
};

#endif