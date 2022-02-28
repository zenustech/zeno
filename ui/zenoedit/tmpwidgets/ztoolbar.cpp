#include "ztoolbar.h"
#include <comctrl/ztoolbutton.h>
#include <comctrl/zobjectbutton.h>


ZShapeBar::ZShapeBar(QWidget* parent)
    : QWidget(parent)
{
    QHBoxLayout* pLayout = new QHBoxLayout;
    pLayout->setSpacing(10);
    pLayout->addWidget(new ZObjectButton(QIcon(":/icons/sphere.png"), tr("Sphere")));
    pLayout->addWidget(new ZObjectButton(QIcon(":/icons/cube.png"), tr("Cube")));
    pLayout->addWidget(new ZObjectButton(QIcon(":/icons/tube.png"), tr("Tube")));
    pLayout->addWidget(new ZObjectButton(QIcon(":/icons/torus.png"), tr("Torus")));
    pLayout->addWidget(new ZObjectButton(QIcon(":/icons/circle.png"), tr("Circle")));
    pLayout->addWidget(new ZObjectButton(QIcon(":/icons/curve.png"), tr("Curve")));
    pLayout->addWidget(new ZObjectButton(QIcon(":/icons/pathicon.png"), tr("Path")));
    pLayout->addWidget(new ZObjectButton(QIcon(":/icons/spray_paint.png"), tr("Spray Paint")));
    pLayout->addWidget(new ZObjectButton(QIcon(":/icons/Icosahedron.png"), tr("Icosahedron")));
    pLayout->addWidget(new ZObjectButton(QIcon(":/icons/mobius.png"), tr("Mobius")));
    pLayout->addWidget(new ZObjectButton(QIcon(":/icons/openfile.png"), tr("Open File")));
    pLayout->addStretch();
    setLayout(pLayout);

    QPalette palette;
    palette.setBrush(QPalette::Window, QColor(58, 58, 58));
    setPalette(palette);
}

void ZShapeBar::paintEvent(QPaintEvent* e)
{
    QPainter painter(this);
    painter.fillRect(rect(), QColor(69, 69, 69));
}

ZTextureBar::ZTextureBar(QWidget* parent)
    : QWidget(parent)
{
    QHBoxLayout* pLayout = new QHBoxLayout;
    pLayout->setSpacing(10);
    pLayout->addWidget(new ZObjectButton(QIcon(":/icons/splash.png"), tr("Sphere")));
    pLayout->addWidget(new ZObjectButton(QIcon(":/icons/hellow.png"), tr("Cube")));
    pLayout->addWidget(new ZObjectButton(QIcon(":/icons/prism.png"), tr("Tube")));
    pLayout->addWidget(new ZObjectButton(QIcon(":/icons/plane.png"), tr("Torus")));
    pLayout->addWidget(new ZObjectButton(QIcon(":/icons/metalball.png"), tr("Circle")));
    pLayout->addWidget(new ZObjectButton(QIcon(":/icons/halfpipe.png"), tr("Curve")));
    pLayout->addStretch();
    setLayout(pLayout);
}

void ZTextureBar::paintEvent(QPaintEvent* e)
{
	QPainter painter(this);
	painter.fillRect(rect(), QColor(69, 69, 69));
}

ZToolbar::ZToolbar(QWidget* parent)
    : QWidget(parent)
{
    QVBoxLayout* pLayout = new QVBoxLayout;
    pLayout->setSpacing(10);
    pLayout->addWidget(new ZMiniToolButton(QIcon(":/icons/select.png")));
    pLayout->addWidget(new ZMiniToolButton(QIcon(":/icons/lock.png")));
    pLayout->addWidget(new ZMiniToolButton(QIcon(":/icons/move.png")));
    pLayout->addWidget(new ZMiniToolButton(QIcon(":/icons/rotate.png")));
    pLayout->addWidget(new ZMiniToolButton(QIcon(":/icons/scale.png")));
    pLayout->addWidget(new ZMiniToolButton(QIcon(":/icons/magnet.png")));
    pLayout->addWidget(new ZMiniToolButton(QIcon(":/icons/magnet_grid.png")));
    pLayout->addWidget(new ZMiniToolButton(QIcon(":/icons/magnet_point.png")));
    pLayout->addStretch();
    setLayout(pLayout);
}
