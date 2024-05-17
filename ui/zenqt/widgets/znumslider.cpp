#include "znumslider.h"
#include "style/zenostyle.h"
#include "widgets/zlabel.h"
#include <zeno/utils/log.h>


ZNumSlider::ZNumSlider(QVector<qreal> scales, QWidget* parent)
    : QWidget(parent)
    , m_scales(scales)
    , m_currLabel(nullptr)
{
    QVBoxLayout* pLayout = new QVBoxLayout;
    for (auto scale : m_scales)
    {
        ZTextLabel* pLabel = new ZTextLabel(QString::number(scale));
        pLabel->setTextColor(QColor(80, 80, 80));
        pLabel->setProperty("cssClass", "numslider");
        pLabel->setAlignment(Qt::AlignCenter);
        pLabel->setEnterCursor(Qt::SizeHorCursor);
        pLayout->addWidget(pLabel);
        pLabel->installEventFilter(this);
        pLabel->setAttribute(Qt::WA_TranslucentBackground, true);
        m_labels.append(pLabel);
    }
    pLayout->setMargin(0);
    pLayout->setSpacing(0);
    setLayout(pLayout);

    QPalette pal = this->palette();
    pal.setBrush(QPalette::Window, QColor(21, 21, 21));
    setPalette(pal);
}

ZNumSlider::~ZNumSlider()
{

}

void ZNumSlider::paintEvent(QPaintEvent* event)
{
    QWidget::paintEvent(event);
}

void ZNumSlider::focusOutEvent(QFocusEvent* event)
{
    QWidget::focusOutEvent(event);
    hide();
    emit slideFinished();
}

void ZNumSlider::keyReleaseEvent(QKeyEvent* event)
{
    if (isVisible())
    {
        hide();
        emit slideFinished();
    }
    event->accept();
}

void ZNumSlider::mousePressEvent(QMouseEvent* event)
{
    m_lastPos = event->pos();
    activateLabel(mapFromGlobal(event->globalPos()), true);
    QWidget::mousePressEvent(event);
}

void ZNumSlider::mouseMoveEvent(QMouseEvent* event)
{
    QPoint pos = mapFromGlobal(event->globalPos());
    if (activateLabel(pos))
        return;

    if (m_currLabel)
    {
        qreal dx = event->pos().x() - m_lastPos.x();
        static const int speed_factor = 15;
        if (std::abs(dx) > speed_factor)
        {
            qreal scale = m_currLabel->text().toFloat();
            int pieces = dx / speed_factor;
            qreal Dx = (pieces * scale);
            emit numSlided(Dx);
            m_lastPos = event->pos();
        }
    }
    QWidget::mouseMoveEvent(event);
}

void ZNumSlider::mouseReleaseEvent(QMouseEvent* event)
{
	for (auto label : m_labels)
	{
		if (label != m_currLabel)
			label->setTransparent(false);
	}
}

bool ZNumSlider::activateLabel(QPoint& pos, bool pressed)
{
    for (ZTextLabel* label : m_labels)
    {
        if (label)
        {
            if (label->geometry().contains(pos.x(), pos.y()))
            {
                bool bOK = false;
                qreal scale = label->text().toFloat(&bOK);
                if (bOK)
                {
                    if (pressed || label != m_currLabel)
                    {
                        if (m_currLabel)
                        {
                            qApp->sendEvent(m_currLabel, new QEvent(QEvent::Leave));
                        }
                        m_currLabel = label;
                        qApp->sendEvent(m_currLabel, new QEvent(QEvent::Enter));
                        return true;
                    }
                }
            }
        }
    }
    return false;
}
