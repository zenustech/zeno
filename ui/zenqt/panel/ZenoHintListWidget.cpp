#include "ZenoHintListWidget.h"

ZenoHintListWidget::ZenoHintListWidget()
    : QWidget(nullptr)
    , m_listView(new QListView(this))
    , m_model(new QStringListModel(this))
    , m_currentLineEdit(nullptr)
{
    setMinimumSize({ minWidth,minHeight });
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
    resetCurrentItem();
};

void ZenoHintListWidget::resetCurrentItem()
{
    QItemSelectionModel* selModel = m_listView->selectionModel();
    const QModelIndex& first = m_model->index(0, 0);
    if (first.isValid())
        selModel->setCurrentIndex(first, QItemSelectionModel::SelectCurrent);
}

void ZenoHintListWidget::clearCurrentItem() {
    QItemSelectionModel* selModel = m_listView->selectionModel();
    selModel->clearCurrentIndex();
    selModel->clearSelection();
};

void ZenoHintListWidget::resetSize()
{
    setGeometry(x(), y(), minWidth, minHeight);
    m_button->move(width() - SideLength, height() - SideLength);
}

void ZenoHintListWidget::setCurrentZlineEdit(ZLineEdit* linedit)
{
    m_currentLineEdit = linedit;
}

QString ZenoHintListWidget::getCurrentText()
{
    QItemSelectionModel* selModel = m_listView->selectionModel();
    const QModelIndex& curr = selModel->currentIndex();
    if (curr.isValid())
        return curr.data(Qt::DisplayRole).toString();
    return "";
}

void ZenoHintListWidget::setCalcPropPanelPosFunc(std::function<QPoint()> func)
{
    m_getPropPanelPosfunc = func;
}

QPoint ZenoHintListWidget::getPropPanelPos()
{
    return m_getPropPanelPosfunc();
}

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
                    this->hide();
                    emit escPressedHide();
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
                if (QWidget* wid = qobject_cast<QWidget*>(watched)) //点击区域不在内部则hide
                {
                    const QPoint& globalPos = wid->mapToGlobal(e->pos());
                    const QPoint& lefttop = mapToGlobal(QPoint(0, 0));
                    const QPoint& rightbottom = mapToGlobal(QPoint(width(), height()));
                    if (globalPos.x() < lefttop.x() || globalPos.x() > rightbottom.x() || globalPos.y() < lefttop.y() || globalPos.y() > rightbottom.y())
                    {
                        hide();
                        emit clickOutSideHide(wid);
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

void ZenoHintListWidget::mouseReleaseEvent(QMouseEvent* event)
{
    if (m_resizing)
    {
        emit resizeFinished();
    }
    QWidget::mouseReleaseEvent(event);
}

void ZenoHintListWidget::paintEvent(QPaintEvent* event)
{
    QWidget::paintEvent(event);
    QPainter painter(this);
    painter.fillRect(rect(), "#22252C");
    painter.setPen(Qt::black);
    painter.drawRect(QRect(0, 0, width() - 1, height() - 1));
    QWidget::paintEvent(event);
}

ZenoFuncDescriptionLabel::ZenoFuncDescriptionLabel()
    : m_currentFunc("")
{
    setWindowFlags(windowFlags() | Qt::WindowStaysOnTopHint);
    //setMinimumSize({ 100, 50 });
    m_label = new QLabel(this);
    m_label->setStyleSheet("QLabel{ font-size: 10pt; color: rgb(160, 178, 194)}");
    
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->setContentsMargins(10, 10, 10, 10);
    layout->addWidget(m_label);
    setLayout(layout);
    qApp->installEventFilter(this);

    hide();
}

void ZenoFuncDescriptionLabel::setDesc(zeno::FUNC_INFO func, int pos)
{
    std::string txtToSet = func.rettype + " " + func.name + " " + "(";
    for (int i = 0; i < func.args.size(); i++) {
        std::string arg_content = func.args[i].type + " " + func.args[i].name;
        if (i == pos) {
            txtToSet += "<b>" + arg_content + "</b> ";
        }
        else {
            txtToSet += arg_content + " ";
        }
    }
    txtToSet += ")";

    txtToSet = "<p>" + txtToSet + "</p>";
    txtToSet += "<p>" + func.tip + "</p>";
    txtToSet = "<html><head><style> p { margin: 10; } </style></head><body><div>" + txtToSet + "</div></body></html>";
    m_label->setText(QString::fromStdString(txtToSet));
    m_label->setFont(QApplication::font());
    adjustSize();
}

void ZenoFuncDescriptionLabel::setCurrentFuncName(std::string funcName)
{
    m_currentFunc = funcName;
}

std::string ZenoFuncDescriptionLabel::getCurrentFuncName()
{
    return m_currentFunc;
}

void ZenoFuncDescriptionLabel::setCalcPropPanelPosFunc(std::function<QPoint()> func)
{
    m_getPropPanelPosfunc = func;
}

QPoint ZenoFuncDescriptionLabel::getPropPanelPos()
{
    return m_getPropPanelPosfunc();
}

bool ZenoFuncDescriptionLabel::eventFilter(QObject* watched, QEvent* event)
{
    if (this->isVisible())
    {
        if (event->type() == QEvent::MouseButtonPress)
        {
            if (QMouseEvent* e = static_cast<QMouseEvent*>(event))
            {
                if (QWidget* wid = qobject_cast<QWidget*>(watched)) //点击区域不在内部则hide
                {
                    const QPoint& globalPos = wid->mapToGlobal(e->pos());
                    const QPoint& lefttop = mapToGlobal(QPoint(0, 0));
                    const QPoint& rightbottom = mapToGlobal(QPoint(width(), height()));
                    if (globalPos.x() < lefttop.x() || globalPos.x() > rightbottom.x() || globalPos.y() < lefttop.y() || globalPos.y() > rightbottom.y())
                    {
                        hide();
                    }
                }
            }
        }
    }
    return QWidget::eventFilter(watched, event);
}

void ZenoFuncDescriptionLabel::paintEvent(QPaintEvent* event)
{
    QWidget::paintEvent(event);
    QPainter painter(this);
    painter.fillRect(rect(), "#22252C");
    painter.setPen(Qt::black);
    painter.drawRect(QRect(0, 0, width() - 1, height() - 1));
    QWidget::paintEvent(event);
}
