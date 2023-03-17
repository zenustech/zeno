#if 0

#include "zenodocktitlewidget.h"
#include <comctrl/ziconbutton.h>
#include <comctrl/ztoolbutton.h>
#include <zenoui/style/zenostyle.h>
#include <zenomodel/include/igraphsmodel.h>
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "viewport/viewportwidget.h"
#include <zenomodel/include/graphsmanagment.h>
#include "viewport/zenovis.h"
#include "util/log.h"
#include <QFileDialog>
#include <zenovis/ObjectsManager.h>
#include <zeno/types/CameraObject.h>

namespace zenovis {
    struct Camera {
        void setCamera(zeno::CameraData const &cam);
    };
} // namespace zenovis

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
	pDockSwitchBtn->setBackgroundClr(QColor(), QColor(51, 51, 51), QColor(51, 51, 51), QColor(51, 51, 51));

	ZToolButton* pDockOptionsBtn = new ZToolButton(ZToolButton::Opt_HasIcon, QIcon(":/icons/dockOption.svg"), ZenoStyle::dpiScaledSize(QSize(16, 16)));
	pDockOptionsBtn->setMargins(QMargins(10, 10, 10, 10));
	pDockOptionsBtn->setBackgroundClr(QColor(), QColor(51, 51, 51), QColor(51, 51, 51), QColor(51, 51, 51));

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

void ZenoDockTitleWidget::mouseDoubleClickEvent(QMouseEvent* event)
{
    QWidget::mouseDoubleClickEvent(event);
    emit doubleClicked();
}

void ZenoDockTitleWidget::updateByType(DOCK_TYPE type)
{

}

QAction* ZenoDockTitleWidget::createAction(const QString& text)
{
	QAction* pAction = new QAction(text);
	connect(pAction, &QAction::triggered, this, [=]() {
		emit actionTriggered(pAction);
		});
	return pAction;
}

void ZenoDockTitleWidget::onDockSwitchClicked()
{
	QMenu* menu = new QMenu(this);
    QFont font = zenoApp->font();
    font.setPointSize(10);
	font.setBold(false);
	menu->setFont(font);
	QAction* pSwitchEditor = new QAction("Editor");
	QAction* pSwitchView = new QAction("View");
	QAction* pSwitchNodeParam = new QAction("parameter");
	QAction* pSwitchNodeData = new QAction("data");
	QAction *pSwitchLog = new QAction("logger");
	QAction *pSwitchLights = new QAction("lights");
	menu->addAction(pSwitchEditor);
	menu->addAction(pSwitchView);
	menu->addAction(pSwitchNodeParam);
	menu->addAction(pSwitchNodeData);
	menu->addAction(pSwitchLog);
	menu->addAction(pSwitchLights);
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
	connect(pSwitchLog, &QAction::triggered, this, [=]() {
		emit dockSwitchClicked(DOCK_LOG);
	});
    connect(pSwitchLights, &QAction::triggered, this, [=]() {
		emit dockSwitchClicked(DOCK_LIGHTS);
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
	connect(&*graphsMgr, SIGNAL(modelInited(IGraphsModel*)), this, SLOT(onModelInited(IGraphsModel*)));
}

void ZenoEditorDockTitleWidget::initTitleContent(QHBoxLayout* pHLayout)
{
	QMenuBar* pMenuBar = initMenu();
	pHLayout->addWidget(pMenuBar);
	pHLayout->setAlignment(pMenuBar, Qt::AlignVCenter);
    pHLayout->addStretch();

    m_lblTitle = new QLabel;
    QPalette pal = m_lblTitle->palette();
    pal.setColor(QPalette::WindowText, QColor(255, 255, 255, 128));
    m_lblTitle->setPalette(pal);
    QFont font = zenoApp->font();
    font.setPointSize(10);
    m_lblTitle->setFont(font);

    pHLayout->addWidget(m_lblTitle);
    pHLayout->addStretch();
}

QMenuBar* ZenoEditorDockTitleWidget::initMenu()
{
	QMenuBar* pMenuBar = new QMenuBar(this);
    pMenuBar->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Expanding);

	//QMenu* pAdd = new QMenu(tr("Add"));
	//{
		//pAdd->addAction(createAction(tr("Add Subnet")));
		//pAdd->addAction(createAction(tr("Add Node")));
	//}

	QMenu* pEdit = new QMenu(tr("Edit"));
	{
		pEdit->addAction(createAction(tr("Undo")));
		pEdit->addAction(createAction(tr("Redo")));
		pEdit->addAction(createAction(tr("Collaspe")));
		pEdit->addAction(createAction(tr("Expand")));
        pEdit->addAction(createAction(tr("Easy Subgraph")));
        pEdit->addAction(createAction(tr("Open View")));
        pEdit->addAction(createAction(tr("Clear View")));
	}

	QMenu* pOption = new QMenu(tr("Option"));
    {
        pOption->addAction(createAction(tr("Set NASLOC")));
        pOption->addAction(createAction(tr("Set ZENCACHE")));
    }

	//QMenu* pGo = new QMenu(tr("Go"));
	//{

	//}

	//QMenu* pView = new QMenu(tr("View"));
	//{

	//}

	//QMenu* pHelp = new QMenu(tr("Help"));
	//{

	//}

	//pMenuBar->addMenu(pAdd);
	pMenuBar->addMenu(pEdit);
	pMenuBar->addMenu(pOption);
	//pMenuBar->addMenu(pGo);
	//pMenuBar->addMenu(pView);
	//pMenuBar->addMenu(pHelp);

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
    ZASSERT_EXIT(pModel);
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


ZenoViewDockTitle::ZenoViewDockTitle(QWidget* parent)
	: ZenoDockTitleWidget(parent)
	, m_pSolidMode(nullptr)
	, m_pShadingMode(nullptr)
	, m_pOptixMode(nullptr)
{

}

ZenoViewDockTitle::~ZenoViewDockTitle()
{

}

void ZenoViewDockTitle::initTitleContent(QHBoxLayout* pHLayout)
{
    QMenuBar* pMenuBar = initMenu();
    pHLayout->addWidget(pMenuBar);
    pHLayout->setAlignment(pMenuBar, Qt::AlignVCenter);
    pHLayout->addStretch();
}

QMenuBar* ZenoViewDockTitle::initMenu()
{
    QMenuBar* pMenuBar = new QMenuBar(this);
    pMenuBar->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Expanding);

    QMenu* pDisplay = new QMenu(tr("Display"));
    {
        QAction* pAction = new QAction(tr("Smooth Shading"), this);
        pAction->setShortcut(QKeySequence("F5"));
        pAction->setCheckable(true);
        pAction->setChecked(false);
        pDisplay->addAction(pAction);
        connect(pAction, &QAction::triggered, this,
            [=]() {
                Zenovis::GetInstance().getSession()->set_smooth_shading(pAction->isChecked());
                zenoApp->getMainWindow()->updateViewport();
            });

        pAction = new QAction(tr("Normal Check"), this);
        pAction->setShortcut(QKeySequence("Shift+F5"));
        pAction->setCheckable(true);
        pAction->setChecked(false);
        pDisplay->addAction(pAction);
        connect(pAction, &QAction::triggered, this,
            [=]() {
                Zenovis::GetInstance().getSession()->set_normal_check(pAction->isChecked());
                zenoApp->getMainWindow()->updateViewport();
            });

        pAction = new QAction(tr("Wireframe"), this);
        pAction->setShortcut(QKeySequence("F6"));
        pAction->setCheckable(true);
        pAction->setChecked(false);
        pDisplay->addAction(pAction);
        connect(pAction, &QAction::triggered, this,
            [=]() {
                Zenovis::GetInstance().getSession()->set_render_wireframe(pAction->isChecked());
                zenoApp->getMainWindow()->updateViewport();
            });

        pDisplay->addSeparator();
        pAction = new QAction(tr("Show Grid"), this);
        pAction->setShortcut(QKeySequence("Shift+F6"));
        pAction->setCheckable(true);
        pAction->setChecked(true);
        pDisplay->addAction(pAction);
        connect(pAction, &QAction::triggered, this,
            [=]() {
                Zenovis::GetInstance().getSession()->set_show_grid(pAction->isChecked());
                //todo: need a notify mechanism from zenovis/session.
                zenoApp->getMainWindow()->updateViewport();
            });

        pAction = new QAction(tr("Background Color"), this);
        pDisplay->addAction(pAction);
        connect(pAction, &QAction::triggered, this, [=]() {
            auto [r, g, b] = Zenovis::GetInstance().getSession()->get_background_color();
            auto c = QColor::fromRgbF(r, g, b);
            c = QColorDialog::getColor(c);
            if (c.isValid()) {
                Zenovis::GetInstance().getSession()->set_background_color(c.redF(), c.greenF(), c.blueF());
                zenoApp->getMainWindow()->updateViewport();
            }
            });

        pDisplay->addSeparator();
        m_pSolidMode = new QAction(tr("Solid"), this);
        m_pSolidMode->setShortcut(QKeySequence("F7"));
        m_pSolidMode->setCheckable(true);
        pDisplay->addAction(m_pSolidMode);
        connect(m_pSolidMode, &QAction::triggered, this, [=]() {
            const char *e = "bate";
            Zenovis::GetInstance().getSession()->set_render_engine(e);
            zenoApp->getMainWindow()->updateViewport(QString::fromUtf8(e));

            if (m_pShadingMode) m_pShadingMode->setChecked(false);
            if (m_pOptixMode) m_pOptixMode->setChecked(false);
            m_pSolidMode->setChecked(true);
        });
        m_pSolidMode->setChecked(true);     //default mode.

        m_pShadingMode = new QAction(tr("Shading"), this);
        m_pShadingMode->setShortcut(QKeySequence("Shift+F7"));
        m_pShadingMode->setCheckable(true);
        pDisplay->addAction(m_pShadingMode);
        connect(m_pShadingMode, &QAction::triggered, this, [=]() {
            const char *e = "zhxx";
            Zenovis::GetInstance().getSession()->set_render_engine(e);
            //Zenovis::GetInstance().getSession()->set_enable_gi(false);
            zenoApp->getMainWindow()->updateViewport(QString::fromUtf8(e));

            if (m_pSolidMode) m_pSolidMode->setChecked(false);
            if (m_pOptixMode) m_pOptixMode->setChecked(false);
            m_pShadingMode->setChecked(true);
        });

        m_pOptixMode = new QAction(tr("Optix"), this);
        m_pOptixMode->setShortcut(QKeySequence("F8"));
        m_pOptixMode->setCheckable(true);
        pDisplay->addAction(m_pOptixMode);
        connect(m_pOptixMode, &QAction::triggered, this, [=]() {
            const char *e = "optx";
            Zenovis::GetInstance().getSession()->set_render_engine(e);
            zenoApp->getMainWindow()->updateViewport(QString::fromUtf8(e));

            if (m_pSolidMode) m_pSolidMode->setChecked(false);
            if (m_pShadingMode) m_pShadingMode->setChecked(false);
            m_pOptixMode->setChecked(true);
        });
    }

    QMenu* pRecord = new QMenu(tr("Record"));
    {
        QAction* pAction = new QAction(tr("Screenshot"), this);
        pAction->setShortcut(QKeySequence("F12"));
        pRecord->addAction(pAction);
        connect(pAction, &QAction::triggered, this, [=]() {
            //QString path = QDateTime::currentDateTime().toString(QString("yyyy-dd-MM_hh-mm-ss.png"));
            QString path = QFileDialog::getSaveFileName(nullptr, tr("Path to Save"), "", tr("PNG images(*.png);;JPEG images(*.jpg);;BMP images(*.bmp);;EXR images(*.exr);;HDR images(*.hdr);;"));
            QString ext = QFileInfo(path).suffix();
			if (!path.isEmpty()) {
				Zenovis::GetInstance().getSession()->do_screenshot(path.toStdString(), ext.toStdString());
			}
        });
		pAction = createAction(tr("Record Video"));
        pAction->setShortcut(QKeySequence(("Shift+F12")));
        pRecord->addAction(pAction);
    }

    QMenu* pEnvText = new QMenu(tr("EnvTex"));
    {
		QAction* pAction = new QAction(tr("BlackWhite"), this);
		connect(pAction, &QAction::triggered, this, [=]() {
			//todo
		});
		pEnvText->addAction(pAction);

		pAction = new QAction(tr("Creek"), this);
		connect(pAction, &QAction::triggered, this, [=]() {
			//todo
		});
		pEnvText->addAction(pAction);

		pAction = new QAction(tr("Daylight"), this);
		connect(pAction, &QAction::triggered, this, [=]() {
			//todo
		});
		pEnvText->addAction(pAction);

        pAction = new QAction(tr("Default"), this);
		connect(pAction, &QAction::triggered, this, [=]() {
			//todo
		});
        pEnvText->addAction(pAction);

        pAction = new QAction(tr("Footballfield"), this);
        connect(pAction, &QAction::triggered, this, [=]() {
            //todo
        });
        pEnvText->addAction(pAction);

        pAction = new QAction(tr("Forest"), this);
        connect(pAction, &QAction::triggered, this, [=]() {
            //todo
        });
        pEnvText->addAction(pAction);

        pAction = new QAction(tr("Lake"), this);
        connect(pAction, &QAction::triggered, this, [=]() {
            //todo
        });
        pEnvText->addAction(pAction);

        pAction = new QAction(tr("Sea"), this);
        connect(pAction, &QAction::triggered, this, [=]() {
            //todo
        });
        pEnvText->addAction(pAction);
    }

    QMenu* pCamera = new QMenu(tr("Camera"));
    {
		QAction* pAction = new QAction(tr("Node Camera"), this);
		connect(pAction, &QAction::triggered, this, [=]() {
            int frameid = Zenovis::GetInstance().getSession()->get_curr_frameid();
            auto *scene = Zenovis::GetInstance().getSession()->get_scene();
            for (auto const &[key, ptr]: scene->objectsMan->pairs()) {
                if (key.find("MakeCamera") != std::string::npos && key.find(zeno::format(":{}:", frameid)) != std::string::npos) {
                    auto cam = dynamic_cast<zeno::CameraObject*>(ptr)->get();
                    scene->camera->setCamera(cam);
                    zenoApp->getMainWindow()->updateViewport();
                }
            }
        });
        pCamera->addAction(pAction);
    }

    pMenuBar->addMenu(pDisplay);
    pMenuBar->addMenu(pRecord);
    pMenuBar->addMenu(pEnvText);
    pMenuBar->addMenu(pCamera);

    return pMenuBar;
}


ZenoPropDockTitleWidget::ZenoPropDockTitleWidget(QWidget* parent)
	: ZenoDockTitleWidget(parent)
	, m_title(tr("property"))
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
    QFont font = zenoApp->font();
    font.setPointSize(10);
	p.setFont(font);
	p.drawText(rect(), Qt::AlignCenter, m_title);
}

#endif