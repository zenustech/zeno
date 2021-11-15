#include "layerwidget.h"


LayerTreeView::LayerTreeView(QWidget* parent)
    : QTreeView(parent)
{

}

LayerWidget::LayerWidget(QWidget* parent)
    : QWidget(parent)
{
    QVBoxLayout* pLayout = new QVBoxLayout;
    m_pHeader = new LayerTreeView;
    m_pBody = new LayerTreeView;

    pLayout->addWidget(new QLabel(tr("Layer")));
    pLayout->addWidget(m_pHeader);
    pLayout->addWidget(m_pBody);

    setLayout(pLayout);

    initModel();
    m_pHeader->setModel(m_model);
    m_pHeader->expandAll();
}

void LayerWidget::initModel()
{
    bool bShowExample = false;
    m_model = new QStandardItemModel(this);

    QStandardItem* headerItem = new QStandardItem(QIcon(), "Header");
    {
		QStandardItem* nodenameItem = new QStandardItem(QIcon(), "Node-name");
		QStandardItem* statusItem = new QStandardItem(QIcon(), "Status");
        if (bShowExample)
		{
			QStandardItem* view1Item = new QStandardItem(QIcon(), "View1.svg");
			QStandardItem* mute1Item = new QStandardItem(QIcon(), "Mute1.svg");
			QStandardItem* once1Item = new QStandardItem(QIcon(), "Once1.svg");
			statusItem->appendRow(view1Item);
			statusItem->appendRow(mute1Item);
			statusItem->appendRow(once1Item);
		}
		QStandardItem* controlItem = new QStandardItem(QIcon(), "Control");
        if (bShowExample)
		{
			QStandardItem* collaspeItem = new QStandardItem(QIcon(), "Collapse_0.svg");
			controlItem->appendRow(collaspeItem);
		}
		QStandardItem* displayItem = new QStandardItem(QIcon(), "Display");
        if (bShowExample)
		{
			QStandardItem* genshinItem = new QStandardItem(QIcon(), "Genshin_0.svg");
			displayItem->appendRow(genshinItem);
		}

		QStandardItem* backboardItem = new QStandardItem(QIcon(), "Back-board");
        if (bShowExample)
		{
			QStandardItem* backgroundItem = new QStandardItem(QIcon(), "background.svg");
			backboardItem->appendRow(backgroundItem);
		}
		headerItem->appendRow(nodenameItem);
		headerItem->appendRow(statusItem);
		headerItem->appendRow(controlItem);
		headerItem->appendRow(backboardItem);
    }

    QStandardItem* bodyItem = new QStandardItem(QIcon(), "Body");
    {
        QStandardItem* socketItem = new QStandardItem(QIcon(), "Socket");
        if (bShowExample)
        {
            QStandardItem* firmelementItem = new QStandardItem(QIcon(), "firm_element.svg");
            QStandardItem* socket_text = new QStandardItem(QIcon(), "socket_text");
            socketItem->appendRow(firmelementItem);
            socketItem->appendRow(socket_text);
        }
        QStandardItem* backboardItem = new QStandardItem(QIcon(), "Back-board");
        if (bShowExample)
        {
            QStandardItem* backgroundItem = new QStandardItem(QIcon(), "background.svg");
            backboardItem->appendRow(backgroundItem);
        }
        bodyItem->appendRow(socketItem);
        bodyItem->appendRow(backboardItem);
    }

    m_model->appendRow(headerItem);
    m_model->appendRow(bodyItem);
}