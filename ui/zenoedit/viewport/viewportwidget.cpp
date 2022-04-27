#include "viewportwidget.h"
#include "zenovis.h"
#include "camerakeyframe.h"
#include "timeline/ztimeline.h"
#include "graphsmanagment.h"
#include "model/graphsmodel.h"
#include "launch/corelaunch.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"


CameraControl::CameraControl(QWidget* parent)
    : m_mmb_pressed(false)
    , m_theta(0.)
    , m_phi(0.)
    , m_ortho_mode(false)
    , m_fov(45.)
    , m_radius(5.0)
    , m_res(1, 1)
{
    updatePerspective();
}

void CameraControl::setRes(QVector2D res)
{
    m_res = res;
}

void CameraControl::fakeMousePressEvent(QMouseEvent* event)
{
    if (event->buttons() & Qt::MiddleButton) {
        m_lastPos = event->pos();
    } else if (event->buttons() & Qt::LeftButton) {
        auto flag = zenoApp->getStatus("camera/show_info");
        if (!flag.isNull() && flag.toBool()) {
            float x = (float)event->x() / m_res.x();
            float y = (float)event->y() / m_res.y();

            qDebug() << hitOnFloor(x, y);
        }
    }
}

void CameraControl::fakeMouseMoveEvent(QMouseEvent* event)
{
    if (!(event->buttons() & Qt::MiddleButton))
        return;

    float ratio = QApplication::desktop()->devicePixelRatio();
    float xpos = event->x(), ypos = event->y();
    float dx = xpos - m_lastPos.x(), dy = ypos - m_lastPos.y();
    dx *= ratio / m_res[0];
    dy *= ratio / m_res[1];
    bool shift_pressed = event->modifiers() & Qt::ShiftModifier;
    if (shift_pressed)
    {
        float cos_t = cos(m_theta);
        float sin_t = sin(m_theta);
        float cos_p = cos(m_phi);
        float sin_p = sin(m_phi);
        QVector3D back(cos_t * sin_p, sin_t, -cos_t * cos_p);
        QVector3D up(-sin_t * sin_p, cos_t, sin_t * cos_p);
        QVector3D right = QVector3D::crossProduct(up, back);
        up = QVector3D::crossProduct(back, right);
        right.normalize();
        up.normalize();
        QVector3D delta = right * dx + up * dy;
        m_center += delta * m_radius;
    }
    else
    {
        m_theta -= dy * M_PI;
        m_phi += dx * M_PI;
    }
    m_lastPos = QPointF(xpos, ypos);
    updatePerspective();
    showInfo();
}

void CameraControl::updatePerspective()
{
    float cx = m_center[0], cy = m_center[1], cz = m_center[2];
    Zenovis::GetInstance().updatePerspective(m_res, PerspectiveInfo(cx, cy, cz, m_theta, m_phi, m_radius, m_fov, m_ortho_mode));
}

void CameraControl::fakeWheelEvent(QWheelEvent* event)
{
    int dy = event->angleDelta().y();
    float scale = (dy >= 0) ? 0.89 : 1 / 0.89;
    bool shift_pressed = event->modifiers() & Qt::ShiftModifier;
    if (shift_pressed)
        m_fov /= scale;
    m_radius *= scale;
    updatePerspective();
    showInfo();
}

void CameraControl::setKeyFrame()
{
    //todo
}

void CameraControl::showInfo() {
    if (zenoApp->getStatus("camera/show_info").toBool()) {
        auto p = realPos();
        float cos_t = cos(m_theta);
        float sin_t = sin(m_theta);
        float cos_p = cos(m_phi);
        float sin_p = sin(m_phi);
        QVector3D back(cos_t * sin_p, sin_t, -cos_t * cos_p);
        QVector3D up(-sin_t * sin_p, cos_t, sin_t * cos_p);
        QVector3D right = QVector3D::crossProduct(up, back);
        up = QVector3D::crossProduct(back, right);
        right.normalize();
        up.normalize();
        qDebug() << QString("real pos: (%1, %2, %3), up: (%4, %5, %6), right: (%7, %8, %9), fov: %10")
            .arg(p.x())
            .arg(p.y())
            .arg(p.z())
            .arg(up.x())
            .arg(up.y())
            .arg(up.z())
            .arg(right.x())
            .arg(right.y())
            .arg(right.z())
            .arg(m_fov)
        ;
    }
}

QVector3D CameraControl::realPos() const {
    float cos_t = std::cos(m_theta);
    float sin_t = std::sin(m_theta);
    float cos_p = std::cos(m_phi);
    float sin_p = std::sin(m_phi);
    QVector3D back(cos_t * sin_p, sin_t, -cos_t * cos_p);
    return m_center - back * m_radius;
}

// x, y from [0, 1]
QVector3D CameraControl::screenToWorldRay(float x, float y) const {
    float cos_t = cos(m_theta);
    float sin_t = sin(m_theta);
    float cos_p = cos(m_phi);
    float sin_p = sin(m_phi);
    QVector3D back(cos_t * sin_p, sin_t, -cos_t * cos_p);
    QVector3D up(-sin_t * sin_p, cos_t, sin_t * cos_p);
    QVector3D right = QVector3D::crossProduct(up, back);
    up = QVector3D::crossProduct(back, right);
    right.normalize();
    up.normalize();
    QMatrix4x4 view;
    view.setToIdentity();
    view.lookAt(realPos(), m_center, up);
    x = (x - 0.5) * 2;
    y = (y - 0.5) * (-2);
    float v = std::tan(m_fov * M_PI / 180.f * 0.5f);
    float aspect = res().x() / res().y();
    auto dir = QVector3D(v * x * aspect, v * y, -1);
    dir = dir.normalized();
    dir = view.inverted().mapVector(dir);
    return dir;
}

QVariant CameraControl::hitOnFloor(float x, float y) const {
    auto dir = screenToWorldRay(x, y);
    auto pos = realPos();
    float t = (0 - pos.y()) / dir.y();
    if (t > 0) {
        auto p = pos + dir * t;
        return p;
    } else {
        return {};
    }
}

ViewportWidget::ViewportWidget(QWidget* parent)
    : QGLWidget(parent)
    , m_camera(nullptr)
{
    QGLFormat fmt;
    int nsamples = 16;  // TODO: adjust in a zhouhang-panel
    fmt.setSamples(nsamples);
    fmt.setVersion(3, 2);
    fmt.setProfile(QGLFormat::CoreProfile);
    setFormat(fmt);

    m_camera = std::make_shared<CameraControl>();
    Zenovis::GetInstance().m_camera_control = m_camera.get();
}

ViewportWidget::~ViewportWidget()
{
}

namespace {
struct OpenGLProcAddressHelper {
    inline static QGLContext *ctx;

    static void *getProcAddress(const char *name) {
        return (void *)ctx->getProcAddress(name);
    }
};
}

void ViewportWidget::initializeGL()
{
    OpenGLProcAddressHelper::ctx = context();
    Zenovis::GetInstance().loadGLAPI((void *)OpenGLProcAddressHelper::getProcAddress);
    Zenovis::GetInstance().initializeGL();
}

void ViewportWidget::resizeGL(int nx, int ny)
{
    float ratio = QApplication::desktop()->devicePixelRatio();
    m_camera->setRes(QVector2D(nx * ratio, ny * ratio));
    m_camera->updatePerspective();
}

void ViewportWidget::paintGL()
{
    Zenovis::GetInstance().paintGL();
    checkRecord();
}

void ViewportWidget::checkRecord()
{
    int f = Zenovis::GetInstance().getCurrentFrameId();
    if (!record_path.empty() /*&& f <= frame_end*/) //py has bug: frame_end not initialized.
    {
        QVector2D oldRes = m_camera->res();
        m_camera->setRes(record_res);
        m_camera->updatePerspective();
        Zenovis::GetInstance().recordGL(record_path);
        m_camera->setRes(oldRes);
        m_camera->updatePerspective();
        //if f == self.frame_end:
        //    self.parent_widget.record_video.finish_record()
    }
}

void ViewportWidget::mousePressEvent(QMouseEvent* event)
{
    _base::mousePressEvent(event);
    m_camera->fakeMousePressEvent(event);
    update();
}

void ViewportWidget::mouseMoveEvent(QMouseEvent* event)
{
    _base::mouseMoveEvent(event);
    m_camera->fakeMouseMoveEvent(event);
    update();
}

void ViewportWidget::wheelEvent(QWheelEvent* event)
{
    _base::wheelEvent(event);
    m_camera->fakeWheelEvent(event);
    update();
}


QDMDisplayMenu::QDMDisplayMenu()
{
    setTitle(tr("Display"));
    QAction* pAction = new QAction(tr("Show Grid"), this);
    pAction->setCheckable(true);
    pAction->setChecked(true);
    addAction(pAction);

    pAction = new QAction(tr("Background Color"), this);
    addAction(pAction);

    addSeparator();

    pAction = new QAction(tr("Smooth Shading"), this);
    pAction->setCheckable(true);
    pAction->setChecked(false);
    addAction(pAction);

    pAction = new QAction(tr("Wireframe"), this);
    pAction->setCheckable(true);
    pAction->setChecked(false);

    addSeparator();

    pAction = new QAction(tr("Camera Keyframe"), this);
    addAction(pAction);

    addSeparator();

    pAction = new QAction(tr("Use English"), this);
    pAction->setCheckable(true);
    pAction->setChecked(true);
    addAction(pAction);
}

QDMRecordMenu::QDMRecordMenu()
{
    setTitle(tr("Record"));

    QAction* pAction = new QAction(tr("Screenshot"), this);
    pAction->setShortcut(QKeySequence("F12"));
    addAction(pAction);

    pAction = new QAction(tr("Record Video"), this);
    pAction->setShortcut(QKeySequence(tr("Shift+F12")));
    addAction(pAction);
}


DisplayWidget::DisplayWidget(ZenoMainWindow* pMainWin)
    : QWidget(pMainWin)
    , m_view(nullptr)
    , m_timeline(nullptr)
    , m_mainWin(pMainWin)
{
    QVBoxLayout* pLayout = new QVBoxLayout;
    pLayout->setContentsMargins(0, 0, 0, 0);
    pLayout->setSpacing(0);

    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);

    /*
    ZMenuBar* menuBar = new ZMenuBar;
    menuBar->setMaximumHeight(26);

    QDMDisplayMenu* menuDisplay = new QDMDisplayMenu;
    menuBar->addMenu(menuDisplay);
    QDMRecordMenu* recordDisplay = new QDMRecordMenu;
    menuBar->addMenu(recordDisplay);

    pLayout->addWidget(menuBar);
    */

    m_view = new ViewportWidget;
    pLayout->addWidget(m_view);

    m_timeline = new ZTimeline;
    pLayout->addWidget(m_timeline);
    setLayout(pLayout);

    //RecordVideoDialog
    m_camera_keyframe = new CameraKeyframeWidget;
    Zenovis::GetInstance().m_camera_keyframe = m_camera_keyframe;

	connect(&Zenovis::GetInstance(), SIGNAL(frameUpdated(int)), m_timeline, SLOT(onTimelineUpdate(int)));
	connect(m_timeline, SIGNAL(playForward(bool)), &Zenovis::GetInstance(), SLOT(startPlay(bool)));
	connect(m_timeline, SIGNAL(sliderValueChanged(int)), &Zenovis::GetInstance(), SLOT(setCurrentFrameId(int)));
	connect(m_timeline, SIGNAL(run(int, int)), this, SLOT(onRunClicked(int, int)));

	QTimer* pTimer = new QTimer;
	connect(pTimer, SIGNAL(timeout()), this, SLOT(updateFrame()));
	pTimer->start(16);
}

DisplayWidget::~DisplayWidget()
{

}

void DisplayWidget::init()
{
    //m_camera->installEventFilter(this);
}

QSize DisplayWidget::sizeHint() const
{
    return QSize(12, 400);
}

void DisplayWidget::updateFrame()
{
    if (m_mainWin && m_mainWin->inDlgEventLoop())
        return;
    m_view->update();
}

void DisplayWidget::onRunClicked(int beginFrame, int endFrame)
{
	auto pGraphsMgr = zenoApp->graphsManagment();
	IGraphsModel* pModel = pGraphsMgr->currentModel();
    if (!pModel)
        return;
	GraphsModel* pLegacy = qobject_cast<GraphsModel*>(pModel);
	launchProgram(pLegacy, beginFrame, endFrame);
}
