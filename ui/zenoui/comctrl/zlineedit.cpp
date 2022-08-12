#include "zlineedit.h"
#include "znumslider.h"


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

void ZLineEdit::setNumSlider(const QVector<qreal>& steps)
{
    m_steps = steps;
    m_pSlider = new ZNumSlider(m_steps, this);
    m_pSlider->setWindowFlags(Qt::Window | Qt::FramelessWindowHint);
    m_pSlider->hide();

    connect(m_pSlider, &ZNumSlider::numSlided, this, [=](qreal val) {
        bool bOk = false;
        qreal num = this->text().toFloat(&bOk);
        if (bOk)
        {
            num = num + val;
            QString newText = QString::number(num);
            setText(newText);
        }
    });
    connect(m_pSlider, &ZNumSlider::slideFinished, this, [=]() {
        emit editingFinished();
    });
}

void ZLineEdit::mouseReleaseEvent(QMouseEvent* event)
{
    QLineEdit::mouseReleaseEvent(event);
}

void ZLineEdit::popup()
{
    QPoint pos;
    if (QWidget* pWid = parentWidget())
    {
        pos = pWid->mapToGlobal(geometry().center());
        QSize sz = m_pSlider->size();
        pos -= QPoint(sz.width() / 2, sz.height() / 2);
    }
    else
    {
        pos = this->cursor().pos();
    }
    m_pSlider->move(pos);
    m_pSlider->show();
    m_pSlider->activateWindow();
    m_pSlider->raise();
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
