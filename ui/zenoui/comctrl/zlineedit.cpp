#include "zlineedit.h"
#include "zscaleslider.h"


ZLineEdit::ZLineEdit(QWidget* parent)
    : QLineEdit(parent)
    , m_pSlider(nullptr)
{
}

ZLineEdit::ZLineEdit(const QString& text, QWidget* parent)
    : QLineEdit(text, parent)
    , m_pSlider(nullptr)
{
}

void ZLineEdit::setScalesSlider(QVector<qreal> scales)
{
    m_scales = scales;
    m_pSlider = new ZScaleSlider(m_scales, this);
    m_pSlider->setWindowFlags(Qt::Window | Qt::FramelessWindowHint);
    m_pSlider->hide();

    connect(m_pSlider, &ZScaleSlider::numSlided, this, [=](qreal val) {
        bool bOk = false;
        qreal num = this->text().toFloat(&bOk);
        if (bOk)
        {
            num = num + val;
            QString newText = QString::number(num);
            setText(newText);
        }
    });
    connect(m_pSlider, &ZScaleSlider::slideFinished, this, [=]() {
        emit editingFinished();
    });
}

void ZLineEdit::mouseReleaseEvent(QMouseEvent* event)
{
    QLineEdit::mouseReleaseEvent(event);
}

void ZLineEdit::popup()
{
    QRect rect = geometry();
    QPoint bottomLeft = this->mapToGlobal(rect.bottomLeft());
    QPoint pt = this->cursor().pos();

    QObject* parent = this->parent();
    QWidget* parentWid = this->parentWidget();

    m_pSlider->move(pt);
    m_pSlider->show();
    m_pSlider->activateWindow();
    m_pSlider->raise();
    //m_pSlider->setGeometry(QRect(bottomLeft, QSize(rect.width(), 200)));
    //m_pSlider->setFocus();
}

bool ZLineEdit::eventFilter(QObject* watched, QEvent* event)
{
    if (watched == qApp)
    {
        if (event->type() == QEvent::KeyRelease)
        {
            int j;
            j = 0;
        }
    }
    return QLineEdit::eventFilter(watched, event);
}

bool ZLineEdit::event(QEvent* event)
{
    if (event->type() == QEvent::KeyPress)
    {
        QKeyEvent* k = (QKeyEvent*)event;
        if (k->key() == Qt::Key_Alt)
        {
            if (m_pSlider)
            {
                popup();
                k->accept();
                return true;
            }
        }
    }
    return QLineEdit::event(event);
}

void ZLineEdit::keyPressEvent(QKeyEvent* event)
{
    QLineEdit::keyPressEvent(event);
}

void ZLineEdit::keyReleaseEvent(QKeyEvent* event)
{
    int k = event->key();
    if (k == Qt::Key_Alt)
    {
        if (m_pSlider)
            m_pSlider->hide();
    }
    QLineEdit::keyReleaseEvent(event);
}
