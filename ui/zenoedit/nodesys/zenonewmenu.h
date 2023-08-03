#ifndef __ZENO_NEWMENU_H__
#define __ZENO_NEWMENU_H__

#include <QtWidgets>
#include <zenomodel/include/modeldata.h>
#include <QSet>

class ZenoGvLineEdit;
class IGraphsModel;
class SearchResultWidget;

class ZenoNewnodeMenu : public QMenu
{
	Q_OBJECT
public:
	ZenoNewnodeMenu(const QModelIndex& subgIdx, const NODE_CATES& cates, const QPointF& scenePos, QWidget* parent = nullptr);
	~ZenoNewnodeMenu();
	void setEditorFocus();

protected:
	bool eventFilter(QObject* watched, QEvent* event) override;

public slots:
	void onTextChanged(const QString& text);

private:
	QList<QAction*> getCategoryActions(IGraphsModel* pModel, QModelIndex subgIdx, QPointF scenePos);
	void updateSearchView(const QString& filter);

	bool m_preSearchMode;
	const NODE_CATES m_cates;
	const QModelIndex m_subgIdx;
	const QPointF m_scenePos;
	ZenoGvLineEdit* m_searchEdit;
	SearchResultWidget* m_searchView;
	QWidgetAction* m_wactSearchEdit;
	QWidgetAction* m_wactSearchView;
	QMap<QString, QString> m_nodeToCate;
	QList<QString> m_condidates;
	QList<QAction*> m_cateActions;
};

#endif