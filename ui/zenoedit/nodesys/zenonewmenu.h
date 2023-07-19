#ifndef __ZENO_NEWMENU_H__
#define __ZENO_NEWMENU_H__

#include <QtWidgets>
#include <zenomodel/include/modeldata.h>
#include <QSet>

class ZenoGvLineEdit;
class IGraphsModel;

class ZenoNewnodeMenu : public QMenu
{
	Q_OBJECT
public:
    ZenoNewnodeMenu(const QModelIndex& subgIdx, const NODE_CATES& cates, const QPointF& scenePos, const QString& text = "", QWidget* parent = nullptr);
	~ZenoNewnodeMenu();
	void setEditorFocus();

protected:
	bool eventFilter(QObject* watched, QEvent* event) override;

public slots:
	void onTextChanged(const QString& text);

private:
	QList<QAction*> getCategoryActions(IGraphsModel* pModel, QModelIndex subgIdx, const QString& filter, QPointF scenePos);

	const NODE_CATES m_cates;
	const QModelIndex m_subgIdx;
	const QPointF m_scenePos;
	ZenoGvLineEdit* m_searchEdit;
	QWidgetAction* m_pWAction;
	QSet<QString> deprecatedNodes;
};

#endif