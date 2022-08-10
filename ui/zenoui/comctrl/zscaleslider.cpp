#include "zscaleslider.h"
#include "style/zenostyle.h"
#include <zenoui/comctrl/zlabel.h>
#include <zeno/utils/log.h>


ZScaleSlider::ZScaleSlider(QVector<qreal> scales, QWidget* parent)
    : QWidget(parent)
    , m_scales(scales)
    , m_currScale(-1)
    , m_currLabel(nullptr)
{
    QVBoxLayout* pLayout = new QVBoxLayout;
    for (auto scale : m_scales)
    {
        ZTextLabel* pLabel = new ZTextLabel(QString::number(scale));
        pLabel->setTextColor(QColor(80, 80, 80));
        pLabel->setFixedSize(ZenoStyle::dpiScaledSize(QSize(56, 45)));
        pLayout->addWidget(pLabel);
        pLabel->installEventFilter(this);
    }
    pLayout->setMargin(0);
    setLayout(pLayout);

    QPalette pal = this->palette();
    pal.setBrush(QPalette::Window, QColor(21, 21, 21));
    setPalette(pal);
}

ZScaleSlider::~ZScaleSlider()
{

}

void ZScaleSlider::paintEvent(QPaintEvent* event)
{
    QWidget::paintEvent(event);
}

void ZScaleSlider::keyReleaseEvent(QKeyEvent* event)
{
    int k = event->key();
    if (event->modifiers() == Qt::AltModifier)
    {
        int j;
        j = 0;
    }
    if (isVisible())
    {
        hide();
        emit slideFinished();
    }
    event->accept();
}

bool ZScaleSlider::eventFilter(QObject* watched, QEvent* event)
{
    if (event->type() == QEvent::Enter)
    {
        ZTextLabel* pLabel = qobject_cast<ZTextLabel*>(watched);
        if (pLabel)
        {
            bool bOK = false;
            qreal scale = pLabel->text().toFloat(&bOK);
            if (bOK)
            {
                m_currScale = scale;
                m_currLabel = pLabel;
            }
        }
    }
    return QWidget::eventFilter(watched, event);
}

void ZScaleSlider::mousePressEvent(QMouseEvent* event)
{
    QWidget::mousePressEvent(event);
    m_lastPos = event->pos();
}

void ZScaleSlider::mouseMoveEvent(QMouseEvent* event)
{
    QPointF pos = event->pos();
    if (m_currLabel)
    {
        qreal dx = pos.x() - m_lastPos.x();
        qreal scale = m_currLabel->text().toFloat();
        int pieces = dx;
        qreal Dx = pieces * scale;
        emit numSlided(Dx);
    }
    m_lastPos = event->pos();
    QWidget::mouseMoveEvent(event);
}
