#include "ZenoHintListWidget.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"

ZenoHintListWidget::ZenoHintListWidget()
    : QWidget(zenoApp->getMainWindow())
    , m_listView(new QListView(this))
    , m_model(new QStringListModel(this))
{
    setMinimumSize({ 80,150 });
    setWindowFlags(windowFlags() | Qt::FramelessWindowHint| Qt::WindowStaysOnTopHint);

    QVBoxLayout* pLayout = new QVBoxLayout(this);
    if (pLayout)
    {
        pLayout->addWidget(m_listView);
        pLayout->setContentsMargins(SideLength/2, SideLength/2, SideLength/2, SideLength/2);
    }
    setLayout(pLayout);

    m_button = new TriangleButton("", this);
    m_button->installEventFilter(this);
    m_button->move(width() - SideLength, height() - SideLength);

    m_listView->installEventFilter(this);
    m_listView->setModel(m_model);
    m_listView->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_listView->setStyleSheet("border: 0px;");
    connect(m_listView, SIGNAL(doubleClicked(const QModelIndex&)), this, SLOT(sltItemSelect(const QModelIndex&)));

    qApp->installEventFilter(this);
    this->hide();
};

void ZenoHintListWidget::setData(QStringList items) {
    m_model->setStringList(items);
};

void ZenoHintListWidget::setActive() {
    m_listView->setFocus();
    QItemSelectionModel* selModel = m_listView->selectionModel();
    const QModelIndex& first = m_model->index(0, 0);
    if (first.isValid())
        selModel->setCurrentIndex(first, QItemSelectionModel::SelectCurrent);
};

void ZenoHintListWidget::clearCurrentItem() {
    QItemSelectionModel* selModel = m_listView->selectionModel();
    selModel->clearCurrentIndex();
    selModel->clearSelection();
};

void ZenoHintListWidget::sltItemSelect(const QModelIndex& selectedIdx) {
    if (selectedIdx.isValid())
    {
        this->hide();
        emit hintSelected(selectedIdx.data(Qt::DisplayRole).toString());
    }
    else {
        QItemSelectionModel* selModel = m_listView->selectionModel();
        auto currentIdex = selModel->currentIndex();
        if (currentIdex.isValid()) {
            this->hide();
            emit hintSelected(selModel->currentIndex().data(Qt::DisplayRole).toString());
        }
    }
};

bool ZenoHintListWidget::eventFilter(QObject* watched, QEvent* event)
{
    if (watched == m_button)
    {
        if (event->type() == QEvent::Enter)
        {
            if (!m_resizing)
            {
                m_resizing = !m_resizing;
                setCursor(Qt::SizeFDiagCursor);
                return true;
            }
        }
        else if (event->type() == QEvent::Leave)
        {
            if (m_resizing)
            {
                m_resizing = !m_resizing;
                setCursor(Qt::ArrowCursor);
                return true;
            }
        }
    }
    if (watched == m_listView)
    {
        if (event->type() == QEvent::KeyPress)
        {
            if (QKeyEvent* e = static_cast<QKeyEvent*>(event))
            {
                if (e->key() == Qt::Key_Enter || e->key() == Qt::Key_Return)
                {
                    sltItemSelect(QModelIndex());
                    return true;
                }
                else if (e->key() == Qt::Key_Escape)
                {
                    emit escPressedHide();
                    this->hide();
                    return true;
                }
            }
        }
    }
    if (this->isVisible())
    {
        if (event->type() == QEvent::MouseButtonPress)
        {
            if (QMouseEvent* e = static_cast<QMouseEvent*>(event))
            {
                if (ZLineEdit* edit = qobject_cast<ZLineEdit*>(watched))
                {
                    if (edit->hasFocus())
                    {
                        hide();
                        return true;
                    }
                }
                else if (QWidget* wid = qobject_cast<QWidget*>(watched)) //点击区域不在内部则hide
                {
                    const QPoint& globalPos = wid->mapToGlobal(e->pos());
                    const QPoint& lefttop = mapToGlobal(QPoint(0, 0));
                    const QPoint& rightbottom = mapToGlobal(QPoint(width(), height()));
                    if (globalPos.x() < lefttop.x() || globalPos.x() > rightbottom.x() || globalPos.y() < lefttop.y() || globalPos.y() > rightbottom.y())
                    {
                        hide();
                        clickOutSideHide();
                    }
                }
            }
        }
    }
    return QWidget::eventFilter(watched, event);
}

void ZenoHintListWidget::mouseMoveEvent(QMouseEvent* event)
{
    if (m_resizing)
    {
        setGeometry(x(), y(), event->pos().x(), event->pos().y());
        m_button->move(m_listView->width(), m_listView->height());
    }
}

void ZenoHintListWidget::paintEvent(QPaintEvent* event)
{
    QWidget::paintEvent(event);
    QPainter painter(this);
    painter.fillRect(rect(), "#22252C");
    painter.setPen(Qt::black);
    painter.drawRect(QRect(0, 0, width() - 1, height() - 1));
}
