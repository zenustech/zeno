#include "zenoopenpathpanel.h"


ZenoOpenPathPanel::ZenoOpenPathPanel(QWidget* parent)
    : QWidget(parent)
{
    initUI();
}

void ZenoOpenPathPanel::initUI()
{
    m_pLayout = new QVBoxLayout(this);
    m_pFileDialog = new QFileDialog(this, "Open path", "", "All Files(*);; ");
    m_pFileDialog->setWindowFlags(Qt::Widget);
    m_pLayout->addWidget(m_pFileDialog);
    m_pFileDialog->installEventFilter(this);
}

bool ZenoOpenPathPanel::eventFilter(QObject* obj, QEvent* evt)
{
    if (obj == m_pFileDialog)
    {
        if (evt->type() == QEvent::ShortcutOverride)
        {
            QKeyEvent* keyEvt = dynamic_cast<QKeyEvent*>(evt);
            if (keyEvt == QKeySequence::Copy) {
                m_pFileDialog->setFocus();
                evt->accept();
                return true;
            }
        }
        else if (evt->type() == QEvent::KeyPress)
        {
            QKeyEvent* keyEvt = dynamic_cast<QKeyEvent*>(evt);
            if (keyEvt == QKeySequence::Copy) {
                QMimeData* pMimeData = new QMimeData;
                const QUrl& url = m_pFileDialog->selectedUrls().value(0);
                QString text;
                if (url.isLocalFile() || url.isEmpty())
                    text = url.toLocalFile();
                else
                    text = url.toString();
                pMimeData->setText(text);
                QApplication::clipboard()->setMimeData(pMimeData);
                evt->accept();
                return true;
            }
        }
        else if (evt->type() == QEvent::HideToParent)
        {
            m_pFileDialog->show();
        }
    }
    return QWidget::eventFilter(obj, evt);
}