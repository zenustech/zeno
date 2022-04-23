#include "zenodocktitlewidget.h"
#include <comctrl/ziconbutton.h>
#include <comctrl/ztoolbutton.h>
#include <zenoui/style/zenostyle.h>
#include <zenoui/include/igraphsmodel.h>
#include "zenoapplication.h"
#include "graphsmanagment.h"


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

	ZToolButton* pDockSwitchBtn = new ZToolButton(ZToolButton::Opt_HasIcon, QIcon(":/icons/ic_layout_container.svg"), ZenoStyle::dpiScaledSize(QSize(16, 16)));
	pDockSwitchBtn->setMargins(QMargins(10, 10, 10, 10));
	pDockSwitchBtn->setBackgroundClr(QColor(51, 51, 51), QColor(51, 51, 51), QColor(51, 51, 51));

	ZToolButton* pDockOptionsBtn = new ZToolButton(ZToolButton::Opt_HasIcon, QIcon(":/icons/dockOption.svg"), ZenoStyle::dpiScaledSize(QSize(16, 16)));
	pDockOptionsBtn->setMargins(QMargins(10, 10, 10, 10));
	pDockOptionsBtn->setBackgroundClr(QColor(51, 51, 51), QColor(51, 51, 51), QColor(51, 51, 51));

	pHLayout->addWidget(pDockSwitchBtn);

	initTitleContent(pHLayout);

	pHLayout->addWidget(pDockOptionsBtn);
	pHLayout->setContentsMargins(0, 0, 0, 0);
	pHLayout->setMargin(0);

	pLayout->addLayout(pHLayout);

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
	, m_lblTitle(nullptr)
{

}

ZenoEditorDockTitleWidget::~ZenoEditorDockTitleWidget()
{

}

void ZenoEditorDockTitleWidget::initModel()
{
	IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
	if (pModel)
		setTitle(pModel->fileName());

	auto graphsMgr = zenoApp->graphsManagment();
	connect(graphsMgr.get(), SIGNAL(modelInited(IGraphsModel*)), this, SLOT(onModelInited(IGraphsModel*)));
}

void ZenoEditorDockTitleWidget::initTitleContent(QHBoxLayout* pHLayout)
{
	pHLayout->addWidget(initMenu());
    pHLayout->addStretch();

    m_lblTitle = new QLabel;
    QPalette pal = m_lblTitle->palette();
    pal.setColor(QPalette::WindowText, QColor(255, 255, 255, 128));
    m_lblTitle->setPalette(pal);
    m_lblTitle->setFont(QFont("HarmonyOS Sans", 11));

    pHLayout->addWidget(m_lblTitle);
    pHLayout->addStretch();
}

QAction* ZenoEditorDockTitleWidget::createAction(const QString& text)
{
	QAction* pAction = new QAction(text);
	connect(pAction, &QAction::triggered, this, [=]() {
		emit actionTriggered(qobject_cast<QAction*>(sender()));
		});
	return pAction;
}

QMenuBar* ZenoEditorDockTitleWidget::initMenu()
{
	QMenuBar* pMenuBar = new QMenuBar(this);
    pMenuBar->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Preferred);

	QMenu* pAdd = new QMenu(tr("Add"));
	{
		pAdd->addAction(createAction(tr("Add Subnet")));
		pAdd->addAction(createAction(tr("Add Node")));
	}

	QMenu* pEdit = new QMenu(tr("Edit"));
	{
		pEdit->addAction(createAction(tr("Undo")));
		pEdit->addAction(createAction(tr("Redo")));
		pEdit->addAction(createAction(tr("Collaspe")));
		pEdit->addAction(createAction(tr("Expand")));
        pEdit->addAction(createAction(tr("Easy Subgraph")));
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

void ZenoEditorDockTitleWidget::setTitle(const QString& title)
{
    m_lblTitle->setText(title);
	update();
}

void ZenoEditorDockTitleWidget::onModelInited(IGraphsModel* pModel)
{
	const QString& fn = pModel->fileName();
	if (fn.isEmpty())
	{
		m_lblTitle->setText("newFile");
	}
	else
	{
		m_lblTitle->setText(fn);
	}

	connect(pModel, SIGNAL(modelClear()), this, SLOT(onModelClear()));
	connect(pModel, SIGNAL(pathChanged(const QString&)), this, SLOT(onPathChanged(const QString&)));
	connect(pModel, SIGNAL(dirtyChanged()), this, SLOT(onDirtyChanged()));
	update();
}

void ZenoEditorDockTitleWidget::onModelClear()
{
    m_lblTitle->setText("");
	update();
}

void ZenoEditorDockTitleWidget::onDirtyChanged()
{
	IGraphsModel* pModel = qobject_cast<IGraphsModel*>(sender());
	Q_ASSERT(pModel);
	bool bDirty = pModel->isDirty();
	QString name = pModel->fileName();
	if (name.isEmpty())
		name = "newFile";
	QString title;
	if (bDirty)
	{
		title = name + "*";
	}
	else
	{
		title = name;
	}
	m_lblTitle->setText(title);
	update();
}

void ZenoEditorDockTitleWidget::onPathChanged(const QString& newPath)
{
	QFileInfo fi(newPath);
	QString fn;
	if (fi.isFile())
		fn = fi.fileName();
	m_lblTitle->setText(fn);
	update();
}

void ZenoEditorDockTitleWidget::paintEvent(QPaintEvent* event)
{
	ZenoDockTitleWidget::paintEvent(event);
}



ZenoPropDockTitleWidget::ZenoPropDockTitleWidget(QWidget* parent)
	: ZenoDockTitleWidget(parent)
	, m_title("property")
{

}

ZenoPropDockTitleWidget::~ZenoPropDockTitleWidget()
{
}

void ZenoPropDockTitleWidget::setTitle(const QString& title)
{
	m_title = title;
	update();
}

void ZenoPropDockTitleWidget::paintEvent(QPaintEvent* event)
{
	ZenoDockTitleWidget::paintEvent(event);

	QPainter p(this);
	p.setPen(QPen(QColor(255, 255, 255, 128)));
	p.setFont(QFont("HarmonyOS Sans", 11));
	p.drawText(rect(), Qt::AlignCenter, m_title);
}