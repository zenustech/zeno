#ifndef __ZENO_SUBNET_PANEL_H__
#define __ZENO_SUBNET_PANEL_H__

#include <QtWidgets>

class ZenoSubnetListView;
class ZenoSubnetTreeView;
class IGraphsModel;
class ZTextLabel;

class ZenoSubnetPanel : public QWidget
{
	Q_OBJECT
public:
	ZenoSubnetPanel(QWidget* parent = nullptr);
	void initModel(IGraphsModel* pModel);
	QSize sizeHint() const override;
	void setViewWay(bool bListView);

signals:
	void clicked(const QModelIndex& index);
	void graphToBeActivated(const QString&);

private slots:
	void onNewSubnetBtnClicked();
	void onModelReset();

protected:
	void paintEvent(QPaintEvent* e) override;

private:
	ZenoSubnetListView* m_pListView;
	ZenoSubnetTreeView* m_pTreeView;
	ZTextLabel* m_pNewSubnetBtn;
	bool m_bListView;
};

#endif