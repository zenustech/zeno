#include "zenodockwidget.h"
#include <QtWidgets/private/qdockwidget_p.h>
#include <comctrl/ziconbutton.h>
#include <style/zenostyle.h>


ZenoDockTitleWidget::ZenoDockTitleWidget(QWidget* parent)
    : QWidget(parent)
{
    QVBoxLayout* pLayout = new QVBoxLayout;
    pLayout->setSpacing(0);
    pLayout->setContentsMargins(0, 0, 0, 0);

    QHBoxLayout* pHLayout = new QHBoxLayout;
    ZIconButton* pDockBtn = new ZIconButton(QIcon(":/icons/dockOption.svg"), ZenoStyle::dpiScaledSize(QSize(36, 36)), QColor(), QColor());
    pHLayout->addStretch();
    pHLayout->addWidget(pDockBtn);
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
}

ZenoDockTitleWidget::~ZenoDockTitleWidget()
{
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


ZenoDockWidget::ZenoDockWidget(const QString &title, QWidget *parent, Qt::WindowFlags flags)
    : _base(title, parent, flags)
{
    init();
}

ZenoDockWidget::ZenoDockWidget(QWidget *parent, Qt::WindowFlags flags)
    : _base(parent, flags)
{
    init();
}

ZenoDockWidget::~ZenoDockWidget()
{
}

void ZenoDockWidget::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);
    painter.fillRect(rect(), QColor(36, 36, 36));
    _base::paintEvent(event);
}

void ZenoDockWidget::init()
{
    QPalette palette = this->palette();
    palette.setBrush(QPalette::Window, QColor(38, 38, 38));
    palette.setBrush(QPalette::WindowText, QColor());
    setPalette(palette);
    setTitleBarWidget(new ZenoDockTitleWidget);
}