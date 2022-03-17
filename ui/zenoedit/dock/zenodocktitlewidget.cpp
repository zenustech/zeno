#include "zenodocktitlewidget.h"
#include <comctrl/ziconbutton.h>
#include <comctrl/ztoolbutton.h>


ZenoDockTitleWidget::ZenoDockTitleWidget(QWidget* parent)
	: QWidget(parent)
{
}

ZenoDockTitleWidget::~ZenoDockTitleWidget()
{
}

void ZenoDockTitleWidget::setupUi()
{
	QVBoxLayout* pLayout = new QVBoxLayout;
	pLayout->setSpacing(0);
	pLayout->setContentsMargins(0, 0, 0, 0);

	QHBoxLayout* pHLayout = new QHBoxLayout;

	ZToolButton* pDockSwitchBtn = new ZToolButton(ZToolButton::Opt_HasIcon, QIcon(":/icons/ic_layout_container.svg"), QSize(16, 16));
	pDockSwitchBtn->setMargins(QMargins(10, 10, 10, 10));
	pDockSwitchBtn->setBackgroundClr(QColor(51, 51, 51), QColor(51, 51, 51), QColor(51, 51, 51));

	ZToolButton* pDockOptionsBtn = new ZToolButton(ZToolButton::Opt_HasIcon, QIcon(":/icons/dockOption.svg"), QSize(16, 16));
	pDockOptionsBtn->setMargins(QMargins(10, 10, 10, 10));
	pDockOptionsBtn->setBackgroundClr(QColor(51, 51, 51), QColor(51, 51, 51), QColor(51, 51, 51));

	pHLayout->addWidget(pDockSwitchBtn);

	initTitleContent(pHLayout);

	pHLayout->addWidget(pDockOptionsBtn);
	pHLayout->setContentsMargins(0, 0, 0, 0);
	pHLayout->setMargin(0);

	QFrame* pLine = new QFrame;
	pLine->setFrameShape(QFrame::HLine);
	pLine->setFrameShadow(QFrame::Plain);
	QPalette pal = pLine->palette();
	pal.setBrush(QPalette::WindowText, QColor(36, 36, 36));
	pLine->setPalette(pal);
	pLine->setFixedHeight(1);       //dpi scaled?
	pLine->setLineWidth(1);

	pLayout->addLayout(pHLayout);
	pLayout->addWidget(pLine);

	setLayout(pLayout);

	connect(pDockOptionsBtn, SIGNAL(clicked()), this, SIGNAL(dockOptionsClicked()));
	connect(pDockSwitchBtn, SIGNAL(clicked()), this, SLOT(onDockSwitchClicked()));
}

void ZenoDockTitleWidget::initTitleContent(QHBoxLayout* pHLayout)
{
	pHLayout->addStretch();
}

QSize ZenoDockTitleWidget::sizeHint() const
{
	QSize sz = QWidget::sizeHint();
	return sz;
}

void ZenoDockTitleWidget::paintEvent(QPaintEvent* event)
{
	QPainter painter(this);
	painter.fillRect(rect(), QColor(58, 58, 58));
	QPen pen(QColor(44, 50, 49), 2);
	painter.setPen(pen);
}

void ZenoDockTitleWidget::updateByType(DOCK_TYPE type)
{

}

void ZenoDockTitleWidget::onDockSwitchClicked()
{
	QMenu* menu = new QMenu(this);
	QFont font("HarmonyOS Sans", 12);
	font.setBold(false);
	menu->setFont(font);
	QAction* pSwitchEditor = new QAction("Editor");
	QAction* pSwitchView = new QAction("View");
	QAction* pSwitchNodeParam = new QAction("parameter");
	QAction* pSwitchNodeData = new QAction("data");
	menu->addAction(pSwitchEditor);
	menu->addAction(pSwitchView);
	menu->addAction(pSwitchNodeParam);
	menu->addAction(pSwitchNodeData);
	connect(pSwitchEditor, &QAction::triggered, this, [=]() {
		emit dockSwitchClicked(DOCK_EDITOR);
		});
	connect(pSwitchView, &QAction::triggered, this, [=]() {
		emit dockSwitchClicked(DOCK_VIEW);
		});
	connect(pSwitchNodeParam, &QAction::triggered, this, [=]() {
		emit dockSwitchClicked(DOCK_NODE_PARAMS);
		});
	connect(pSwitchNodeData, &QAction::triggered, this, [=]() {
		emit dockSwitchClicked(DOCK_NODE_DATA);
		});

	menu->exec(QCursor::pos());
}



ZenoEditorDockTitleWidget::ZenoEditorDockTitleWidget(QWidget* parent)
	: ZenoDockTitleWidget(parent)
{

}

ZenoEditorDockTitleWidget::~ZenoEditorDockTitleWidget()
{

}

void ZenoEditorDockTitleWidget::initTitleContent(QHBoxLayout* pHLayout)
{
	pHLayout->addWidget(initMenu());
	pHLayout->addStretch();
}

QMenuBar* ZenoEditorDockTitleWidget::initMenu()
{
	QMenuBar* pMenuBar = new QMenuBar(this);

	QMenu* pAdd = new QMenu(tr("Add"));
	{
		QAction* pAction = new QAction(tr("Add Subnet"), pAdd);
		pAdd->addAction(pAction);

		pAction = new QAction(tr("Add Node"), pAdd);
		pAdd->addAction(pAction);
	}

	QMenu* pEdit = new QMenu(tr("Edit"));
	{
		QAction* pAction = new QAction(tr("Undo"), pEdit);
		pEdit->addAction(pAction);

		pAction = new QAction(tr("Redo"), pEdit);
		pEdit->addAction(pAction);
	}

	QMenu* pGo = new QMenu(tr("Go"));
	{

	}

	QMenu* pView = new QMenu(tr("View"));
	{

	}

	QMenu* pHelp = new QMenu(tr("Help"));
	{

	}

	pMenuBar->addMenu(pAdd);
	pMenuBar->addMenu(pEdit);
	pMenuBar->addMenu(pGo);
	pMenuBar->addMenu(pView);
	pMenuBar->addMenu(pHelp);

	/* up-right-bottom-left */
	pMenuBar->setStyleSheet(
		"\
    QMenuBar {\
        background-color: transparent;\
        spacing: 3px; \
        color: rgba(255,255,255,0.50);\
    }\
    \
    QMenuBar::item {\
        padding: 10px 8px 7px 8px;\
        background: transparent;\
    }\
    \
    QMenuBar::item:selected {\
        background: #4B9EF4;\
    }\
    \
    QMenuBar::item:pressed {\
        background: #4B9EF4;\
    }\
    "
	);

	return pMenuBar;
}