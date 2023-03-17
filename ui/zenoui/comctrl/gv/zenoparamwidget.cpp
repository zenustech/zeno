#include "zenoparamwidget.h"
#include "zenosocketitem.h"
#include "zgraphicsnumslideritem.h"
#include <zenoui/render/common_id.h>
#include <zenoui/style/zenostyle.h>
#include <zeno/utils/log.h>
#include <zenomodel/include/uihelper.h>
#include "../view/zcomboboxitemdelegate.h"
#include "zenoedit/zenoapplication.h"


ZenoParamWidget::ZenoParamWidget(QGraphicsItem* parent, Qt::WindowFlags wFlags)
    : QGraphicsProxyWidget(parent, wFlags)
{
}

ZenoParamWidget::~ZenoParamWidget()
{
}


void ZenoParamWidget::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    QGraphicsProxyWidget::paint(painter, option, widget);
}

void ZenoParamWidget::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    QGraphicsProxyWidget::mousePressEvent(event);
}

void ZenoParamWidget::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    QGraphicsProxyWidget::mouseReleaseEvent(event);
}

void ZenoParamWidget::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event)
{
    QGraphicsProxyWidget::mouseDoubleClickEvent(event);
    emit doubleClicked();
}


///////////////////////////////////////////////////////////////////////////////
ZenoFrame::ZenoFrame(QWidget* parent, Qt::WindowFlags f)
    : QFrame(parent, f)
{
	setFrameShape(QFrame::VLine);
    QPalette pal = palette();
    pal.setBrush(QPalette::WindowText, QColor(86, 96, 143));
    setPalette(pal);
    setLineWidth(4);
}

ZenoFrame::~ZenoFrame()
{
}

QSize ZenoFrame::sizeHint() const
{
    QSize sz = QFrame::sizeHint();
    return sz;
}

void ZenoFrame::paintEvent(QPaintEvent* e)
{
    QFrame::paintEvent(e);
}


///////////////////////////////////////////////////////////////////////////////
ZenoGvLineEdit::ZenoGvLineEdit(QWidget* parent)
    : QLineEdit(parent)
{
    setAutoFillBackground(false);
}

void ZenoGvLineEdit::paintEvent(QPaintEvent* e)
{
    QLineEdit::paintEvent(e);
}


////////////////////////////////////////////////////////////////////////////////
ZenoParamLineEdit::ZenoParamLineEdit(const QString &text, PARAM_CONTROL ctrl, LineEditParam param, QGraphicsItem *parent)
    : ZenoParamWidget(parent)
    , m_pSlider(nullptr)
    , m_pLineEdit(nullptr)
{
    m_pLineEdit = new ZLineEdit;
    m_pLineEdit->setText(text);
    m_pLineEdit->setTextMargins(param.margins);
    m_pLineEdit->setPalette(param.palette);
    m_pLineEdit->setFont(param.font);
    if (param.propertyParam.isEmpty())
        param.propertyParam = "zeno2_2_lineedit";
    m_pLineEdit->setProperty("cssClass", param.propertyParam);
    m_pLineEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    setWidget(m_pLineEdit);
    connect(m_pLineEdit, SIGNAL(editingFinished()), this, SIGNAL(editingFinished()));
}

void ZenoParamLineEdit::setValidator(const QValidator* pValidator)
{
    m_pLineEdit->setValidator(pValidator);
}

void ZenoParamLineEdit::setNumSlider(QGraphicsScene* pScene, const QVector<qreal>& steps)
{
    if (!pScene)
        return;

    m_pSlider = new ZGraphicsNumSliderItem(steps, nullptr);
    connect(m_pSlider, &ZGraphicsNumSliderItem::numSlided, this, [=](qreal val) {
        bool bOk = false;
        qreal num = this->text().toFloat(&bOk);
        if (bOk)
        {
            num = num + val;
            QString newText = QString::number(num);
            setText(newText);
        }
        emit editingFinished();
    });
    connect(m_pSlider, &ZGraphicsNumSliderItem::slideFinished, this, [=]() {
        m_pLineEdit->setShowingSlider(false);
        emit editingFinished();
    });
    m_pSlider->setZValue(1000);
    m_pSlider->hide();
    pScene->addItem(m_pSlider);
}

void ZenoParamLineEdit::setFont(const QFont &font) 
{
    m_pLineEdit->setFont(font);
}

QString ZenoParamLineEdit::text() const
{
    return m_pLineEdit->text();
}

void ZenoParamLineEdit::setText(const QString &text)
{
    m_pLineEdit->setText(text);
}

QGraphicsView* ZenoParamLineEdit::_getFocusViewByCursor()
{
    QPointF cursorPos = this->cursor().pos();
    const auto views = scene()->views();
    Q_ASSERT(!views.isEmpty());
    for (auto view : views)
    {
        QRect rc = view->viewport()->geometry();
        QPoint tl = view->mapToGlobal(rc.topLeft());
        QPoint br = view->mapToGlobal(rc.bottomRight());
        rc = QRect(tl, br);
        if (rc.contains(cursorPos.toPoint()))
        {
            return view;
        }
    }
    return nullptr;
}

void ZenoParamLineEdit::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Shift)
    {
        if (m_pSlider)
        {
            QPointF pos = this->sceneBoundingRect().center();
            QSizeF sz = m_pSlider->boundingRect().size();

            auto view =  _getFocusViewByCursor();
            if (view)
            {
                // it's very difficult to get current viewport, so we get it by current cursor.
                // but when we move the cursor out of the view, we can't get the current view.

                static QRect screen = QApplication::desktop()->screenGeometry();
                static const int _yOffset = ZenoStyle::dpiScaled(20);
                QPointF cursorPos = this->cursor().pos();
                QPoint viewPoint = view->mapFromGlobal(this->cursor().pos());
                const QPointF sceneCursor = view->mapToScene(viewPoint);
                QPointF screenBR = view->mapToScene(view->mapFromGlobal(screen.bottomRight()));
                cursorPos = mapToScene(cursorPos);

                pos.setX(sceneCursor.x());
                pos.setY(std::min(pos.y(), screenBR.y() - sz.height() / 2 - _yOffset) - sz.height() / 2.);
            }
            else
            {
                pos -= QPointF(sz.width() / 2., sz.height() / 2.);
            }

            m_pSlider->setPos(pos);
            m_pSlider->show();
            m_pLineEdit->setShowingSlider(true);
        }
    }
    ZenoParamWidget::keyPressEvent(event);
}

void ZenoParamLineEdit::keyReleaseEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Shift)
    {
        if (m_pSlider)
        {
            m_pSlider->hide();
            m_pLineEdit->setShowingSlider(false);
        }
    }
    ZenoParamWidget::keyReleaseEvent(event);
}


///////////////////////////////////////////////////////////////////////////
ZenoParamPathEdit::ZenoParamPathEdit(const QString& path, PARAM_CONTROL ctrl, LineEditParam param, QGraphicsItem* parent)
    : ZenoParamWidget(parent)
{
    QGraphicsLinearLayout *pLayout = new QGraphicsLinearLayout(Qt::Horizontal);
    m_pLineEdit = new ZenoParamLineEdit(path, ctrl, param);
    pLayout->addItem(m_pLineEdit);
    pLayout->setContentsMargins(0, 0, 0, 0);

    ImageElement elem;
    elem.image = ":/icons/ic_openfile.svg";
    elem.imageHovered = ":/icons/ic_openfile-on.svg";
    elem.imageOn = ":/icons/ic_openfile-on.svg";
    m_openBtn = new ZenoSvgLayoutItem(elem, ZenoStyle::dpiScaledSize(QSize(30, 30)));
    bool isRead = (ctrl == CONTROL_READPATH);
    pLayout->addItem(m_openBtn);
    pLayout->setItemSpacing(0, 0);
    pLayout->setItemSpacing(0, 0);

    this->setLayout(pLayout);

    //connect slot.
    connect(m_pLineEdit, &ZenoParamLineEdit::editingFinished, this, [=]() {
        emit pathValueChanged(m_pLineEdit->text());
    });
    connect(m_openBtn, &ZenoImageItem::clicked, this, &ZenoParamPathEdit::clicked);
}

void ZenoParamPathEdit::setValidator(QValidator* pValidator)
{
    //will override the original.
    //m_pLineEdit->setValidator(pValidator);
}

QString ZenoParamPathEdit::path() const
{
    return m_pLineEdit->text();
}

void ZenoParamPathEdit::setPath(const QString& path)
{
    if (m_pLineEdit->text() != path)
    {
        m_pLineEdit->setText(path);
        emit pathValueChanged(path);
    }
}

///////////////////////////////////////////////////////////////////////////
ZenoParamCheckBox::ZenoParamCheckBox(QGraphicsItem* parent)
    : ZenoParamWidget(parent)
{
    m_pCheckbox = new ZCheckBox;
    m_pCheckbox->setText("");
    m_pCheckbox->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    setWidget(m_pCheckbox);
    connect(m_pCheckbox, SIGNAL(stateChanged(int)), this, SIGNAL(stateChanged(int)));
}

Qt::CheckState ZenoParamCheckBox::checkState() const
{
    return m_pCheckbox->checkState();
}

void ZenoParamCheckBox::setCheckState(Qt::CheckState state)
{
    m_pCheckbox->setCheckState(state);
}

QSizeF ZenoParamCheckBox::sizeHint(Qt::SizeHint which, const QSizeF& constraint) const
{
    QSizeF sz = ZenoParamWidget::sizeHint(which, constraint);
    return sz;
}


///////////////////////////////////////////////////////////////////////////
ZenoVecEditItem::ZenoVecEditItem(const UI_VECTYPE& vec, bool bFloat, LineEditParam param, QGraphicsScene* pScene, QGraphicsItem* parent)
    : ZenoParamWidget(parent)
    , m_bFloatVec(bFloat)
    , m_param(param)
{
    initUI(vec, bFloat, pScene);
}

void ZenoVecEditItem::initUI(const UI_VECTYPE& vec, bool bFloat, QGraphicsScene* pScene)
{
    for (int i = 0; i < m_editors.size(); i++)
    {
        delete m_editors[i];
    }
    m_editors.clear();

    QGraphicsLinearLayout* pLayout = new QGraphicsLinearLayout(Qt::Horizontal);
    pLayout->setContentsMargins(0, 0, 0, 0);
    pLayout->setSpacing(6);
    for (int i = 0; i < vec.size(); i++)
    {
        const QString& numText = QString::number(vec[i]);
        ZenoParamLineEdit* pLineEdit = new ZenoParamLineEdit(numText, CONTROL_FLOAT, m_param);
        pLineEdit->setNumSlider(pScene, UiHelper::getSlideStep("", bFloat ? CONTROL_FLOAT : CONTROL_INT));
        pLayout->addItem(pLineEdit);
        m_editors.append(pLineEdit);
        connect(pLineEdit, SIGNAL(editingFinished()), this, SIGNAL(editingFinished()));
    }
    setLayout(pLayout);
}

bool ZenoVecEditItem::isFloatType() const
{
    return m_bFloatVec;
}

UI_VECTYPE ZenoVecEditItem::vec() const
{
    UI_VECTYPE vec;
    for (auto editor : m_editors)
    {
        if (m_bFloatVec)
        {
            vec.append(editor->text().toFloat());
        }
        else
        {
            vec.append(editor->text().toInt());
        }
    }
    return vec;
}

void ZenoVecEditItem::setVec(const UI_VECTYPE& vec, bool bFloat, QGraphicsScene* pScene)
{
    if (bFloat != m_bFloatVec || vec.size() != m_editors.size())
    {
        initUI(vec, bFloat, pScene);
    }
    else
    {
        for (int i = 0; i < vec.size(); i++)
        {
            m_editors[i]->setText(QString::number(vec[i]));
        }
    }
}

void ZenoVecEditItem::setVec(const UI_VECTYPE& vec)
{
    for (int i = 0; i < vec.size(); i++)
    {
        m_editors[i]->setText(QString::number(vec[i]));
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ZenoParamLabel::ZenoParamLabel(const QString &text, const QFont &font, const QBrush &fill, QGraphicsItem *parent)
    : ZenoParamWidget(parent)
{
    m_label = new QLabel(text);

    QPalette palette;
    palette.setColor(QPalette::WindowText, fill.color());
    m_label->setFont(font);
    m_label->setPalette(palette);
    m_label->setAttribute(Qt::WA_TranslucentBackground);
    m_label->setAutoFillBackground(true);
    setWidget(m_label);
}

void ZenoParamLabel::setAlignment(Qt::Alignment alignment)
{
    m_label->setAlignment(alignment);
}

void ZenoParamLabel::setText(const QString& text)
{
    m_label->setText(text);
}

void ZenoParamLabel::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget*  widget)
{
    painter->fillRect(boundingRect(), QColor(0,0,0));
    ZenoParamWidget::paint(painter, option, widget);
}


////////////////////////////////////////////////////////////////////////////////////
ZComboBoxItemDelegate::ZComboBoxItemDelegate(QObject *parent)
    : QStyledItemDelegate(parent)
{
}

void ZComboBoxItemDelegate::initStyleOption(QStyleOptionViewItem* option, const QModelIndex& index) const
{
    QStyledItemDelegate::initStyleOption(option, index);

    option->backgroundBrush.setStyle(Qt::SolidPattern);
    if (option->state & QStyle::State_MouseOver)
    {
        option->backgroundBrush.setColor(QColor(23, 160, 252));
    }
    else
    {
        option->backgroundBrush.setColor(QColor(58, 58, 58));
    }
}

void ZComboBoxItemDelegate::paint(QPainter* painter, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    QStyleOptionViewItem opt = option;
    initStyleOption(&opt, index);
    painter->fillRect(opt.rect, opt.backgroundBrush);
    painter->setPen(QPen(QColor(210, 203, 197)));
    painter->drawText(opt.rect.adjusted(8, 0, 0, 0), opt.text);
}

QSize ZComboBoxItemDelegate::sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    int w = ((QWidget *) parent())->width();
    return ZenoStyle::dpiScaledSize(QSize(w, 28));
}


ZenoGvComboBox::ZenoGvComboBox(QWidget *parent)
    : QComboBox(parent)
{
}

void ZenoGvComboBox::paintEvent(QPaintEvent *e)
{
    QComboBox::paintEvent(e);
}


ZenoParamComboBox::ZenoParamComboBox(QGraphicsItem* parent)
    : ZenoParamWidget(parent)
{
    m_combobox = new ZComboBox(true);
    m_combobox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_combobox->setItemDelegate(new ZComboBoxItemDelegate(m_combobox));
    setWidget(m_combobox);

    setZValue(ZVALUE_ELEMENT);
    connect(m_combobox, SIGNAL(activated(int)), this, SLOT(onComboItemActivated(int)));
    connect(m_combobox, SIGNAL(beforeShowPopup()), this, SLOT(onBeforeShowPopup()));
    connect(m_combobox, SIGNAL(afterHidePopup()), this, SLOT(onAfterHidePopup()));
}

ZenoParamComboBox::ZenoParamComboBox(const QStringList &items, ComboBoxParam param, QGraphicsItem *parent)
    : ZenoParamWidget(parent)
{
    m_combobox = new ZComboBox(true);
    m_combobox->addItems(items);
    m_combobox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);    
    m_combobox->setEditable(true);

#if 0
    ZLineEdit* pLineEdit = new ZLineEdit(m_combobox);
    pLineEdit->setTextMargins(param.margins);
    pLineEdit->setPalette(param.palette);
    QFont font = zenoApp->font();
    pLineEdit->setFont(font);
    pLineEdit->setProperty("cssClass", "proppanel");
    pLineEdit->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    m_combobox->setLineEdit(pLineEdit);
#endif

    QListView* pComboboxView = qobject_cast<QListView*>(m_combobox->view());
    if (pComboboxView)
        pComboboxView->installEventFilter(this);

    setWidget(m_combobox);

    setZValue(ZVALUE_ELEMENT);
    connect(m_combobox, SIGNAL(activated(int)), this, SLOT(onComboItemActivated(int)));
    connect(m_combobox, SIGNAL(beforeShowPopup()), this, SLOT(onBeforeShowPopup()));
    connect(m_combobox, SIGNAL(afterHidePopup()), this, SLOT(onAfterHidePopup()));
}

bool ZenoParamComboBox::eventFilter(QObject* object, QEvent* event)
{
    if (event->type() == QEvent::Wheel && object == m_combobox->view())
    {
        //when scroll to the bottom of combobox's view, the event will be ignore,
        //and then trigger zoom, which is not convient when activating.
        //see _ZenoSubGraphView::wheelEvent.
        event->setAccepted(true);
        return true;
    }
    return ZenoParamWidget::eventFilter(object, event);
}

void ZenoParamComboBox::setItems(const QStringList& items)
{
    m_combobox->clear();
    m_combobox->addItems(items);
}

void ZenoParamComboBox::onBeforeShowPopup()
{
    setZValue(ZVALUE_POPUPWIDGET);
}

void ZenoParamComboBox::onAfterHidePopup()
{
    setZValue(ZVALUE_ELEMENT);
}

void ZenoParamComboBox::setText(const QString& text)
{
    m_combobox->setCurrentText(text);
}

QString ZenoParamComboBox::text()
{
    return m_combobox->currentText();
}

void ZenoParamComboBox::onComboItemActivated(int index)
{
    // pay attention to the compatiblity of qt!!!
    QString text = m_combobox->itemText(index);
    emit textActivated(text);
}


////////////////////////////////////////////////////////////////////////////////////
ZenoParamPushButton::ZenoParamPushButton(QGraphicsItem* parent)
    : ZenoParamWidget(parent)
    , m_pBtn(nullptr)
{
    m_pBtn = new QPushButton;
    m_pBtn->setProperty("cssClass", "grayButton");
    m_pBtn->setCursor(Qt::PointingHandCursor);
    setWidget(m_pBtn);
    connect(m_pBtn, SIGNAL(clicked()), this, SIGNAL(clicked()));
}

ZenoParamPushButton::ZenoParamPushButton(const QString& name, const QString& qssName, QGraphicsItem* parent)
    : ZenoParamWidget(parent)
    , m_pBtn(nullptr)
{
    m_pBtn = new QPushButton(name);
    m_pBtn->setProperty("cssClass", qssName);
    m_pBtn->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    m_pBtn->setCursor(Qt::PointingHandCursor);
    setWidget(m_pBtn);
    connect(m_pBtn, SIGNAL(clicked()), this, SIGNAL(clicked()));
}

ZenoParamPushButton::ZenoParamPushButton(const QString &name, int width, QSizePolicy::Policy hor, QGraphicsItem *parent)
    : ZenoParamWidget(parent)
    , m_width(width)
    , m_pBtn(nullptr)
{
    m_pBtn = new QPushButton(name);
    m_pBtn->setProperty("cssClass", "grayButton");
    m_pBtn->setCursor(Qt::PointingHandCursor);
    if (hor == QSizePolicy::Fixed)
        m_pBtn->setFixedWidth(width);
    m_pBtn->setSizePolicy(hor, QSizePolicy::Preferred);
    setWidget(m_pBtn);
    connect(m_pBtn, SIGNAL(clicked()), this, SIGNAL(clicked()));
}

void ZenoParamPushButton::setText(const QString& text)
{
    m_pBtn->setText(text);
}


////////////////////////////////////////////////////////////////////////////////////
ZenoParamOpenPath::ZenoParamOpenPath(const QString &filename, QGraphicsItem *parent)
    : ZenoParamWidget(parent)
    , m_path(filename)
{
    QLineEdit *plineEdit = new QLineEdit(filename);
    QPushButton *openBtn = new QPushButton("...");
    QWidget* pWidget = new QWidget;
    QHBoxLayout *pLayout = new QHBoxLayout;
    plineEdit->setReadOnly(true);
    pLayout->addWidget(plineEdit);
    pLayout->addWidget(openBtn);
    pLayout->setMargin(0);
    pWidget->setLayout(pLayout);
    pWidget->setAutoFillBackground(true);
    setWidget(pWidget);
}


//////////////////////////////////////////////////////////////////////////////////////

ZenoParamMultilineStr::ZenoParamMultilineStr(QGraphicsItem* parent)
    : ZenoParamWidget(parent)
    , m_pTextEdit(nullptr)
{
    m_pTextEdit = new QTextEdit;
    setWidget(m_pTextEdit);
    connect(m_pTextEdit, SIGNAL(textChanged()), this, SIGNAL(textChanged()));
    m_pTextEdit->installEventFilter(this);
    m_pTextEdit->setFrameShape(QFrame::NoFrame);
    /*m_pTextEdit->setFont(param.font);*/
    m_pTextEdit->setMinimumSize(ZenoStyle::dpiScaledSize(QSize(256, 228)));

    QPalette pal;
    pal.setColor(QPalette::Base, QColor(25, 29, 33));
    m_pTextEdit->setPalette(pal);
}

ZenoParamMultilineStr::ZenoParamMultilineStr(const QString &value, LineEditParam param, QGraphicsItem *parent)
    : ZenoParamWidget(parent)
    , m_value(value)
    , m_pTextEdit(nullptr)
{
    m_pTextEdit = new QTextEdit;
    setWidget(m_pTextEdit);
    connect(m_pTextEdit, SIGNAL(textChanged()), this, SIGNAL(textChanged()));
    //m_pTextEdit->setProperty("cssClass", "proppanel");
    m_pTextEdit->installEventFilter(this);
    m_pTextEdit->setFrameShape(QFrame::NoFrame);
    m_pTextEdit->setFont(param.font);
    m_pTextEdit->setMinimumSize(ZenoStyle::dpiScaledSize(QSize(256, 228)));
    m_pTextEdit->setText(value);

	//QTextCharFormat format;
    QFont font = zenoApp->font();
    font.setPointSize(10);
    m_pTextEdit->setCurrentFont(font);
    m_pTextEdit->setFont(font);

    //QPalette pal = param.palette;
    //pal.setColor(QPalette::Base, QColor(25, 29, 33));
    //m_pTextEdit->setPalette(pal);
}

void ZenoParamMultilineStr::setText(const QString& text)
{
    m_pTextEdit->setText(text);
}

QString ZenoParamMultilineStr::text() const
{
    return m_pTextEdit->toPlainText();
}

bool ZenoParamMultilineStr::eventFilter(QObject* object, QEvent* event)
{
    if (object == m_pTextEdit && event->type() == QEvent::FocusOut)
    {
        emit editingFinished();
    }
    return ZenoParamWidget::eventFilter(object, event);
}

void ZenoParamMultilineStr::initTextFormat()
{

}


//////////////////////////////////////////////////////////////////////////////////////
ZenoParamBlackboard::ZenoParamBlackboard(const QString& value, LineEditParam param, QGraphicsItem* parent)
    : ZenoParamWidget(parent)
    , m_value(value)
    , m_pTextEdit(nullptr)
{
    m_pTextEdit = new QTextEdit;
    setWidget(m_pTextEdit);
    connect(m_pTextEdit, SIGNAL(textChanged()), this, SIGNAL(textChanged()));
    m_pTextEdit->installEventFilter(this);
    m_pTextEdit->setFrameShape(QFrame::NoFrame);
    m_pTextEdit->setText(value);

    //set line height
    QTextBlockFormat blockFormat;
    blockFormat.setLineHeight(5, QTextBlockFormat::LineDistanceHeight);
    auto textCursor = m_pTextEdit->textCursor();
    textCursor.setBlockFormat(blockFormat);
    m_pTextEdit->setTextCursor(textCursor);
    //set style
    updateStyleSheet(param.font.pointSize());
}

QString ZenoParamBlackboard::text() const
{
    return m_pTextEdit->toPlainText();
}

void ZenoParamBlackboard::setText(const QString& text)
{
    m_pTextEdit->setText(text);
}
void ZenoParamBlackboard::foucusInEdit() 
{
    m_pTextEdit->setFocus();
}

void ZenoParamBlackboard::updateStyleSheet(int fontSize) 
{
    m_pTextEdit->setStyleSheet(
        ZenoStyle::dpiScaleSheet(QString("QTextEdit { background-color: rgb(0, 0, 0); color: rgb(111, 111, 111); "
                                         "padding:16px 5px 5px 5px; font:%1pt \" HarmonyOS Sans \";}")
                                     .arg(fontSize)));
}

bool ZenoParamBlackboard::eventFilter(QObject* object, QEvent* event)
{
    if (object == m_pTextEdit && event->type() == QEvent::FocusOut) {
        emit editingFinished();
    }
    return ZenoParamWidget::eventFilter(object, event);
}

//////////////////////////////////////////////////////////////////////////////////////
ZenoParamSlider::ZenoParamSlider(Qt::Orientation orientation, int value, const SLIDER_INFO &info, QGraphicsItem *parent)
    : ZenoParamWidget(parent) {
    m_pSlider = new QSlider(orientation);
    m_pSlider->setValue(value);
    m_pSlider->setSingleStep(info.step);
    m_pSlider->setRange(info.min, info.max);
    setWidget(m_pSlider);
    updateStyleSheet();

    QObject::connect(m_pSlider, &QSlider::valueChanged, this, &ZenoParamSlider::valueChanged);

    QObject::connect(m_pSlider, &QSlider::sliderPressed, [=]() {
        QPoint pos = QCursor::pos();
        QToolTip::showText(pos, QString("%1").arg(m_pSlider->value()), nullptr);
    });

    QObject::connect(m_pSlider, &QSlider::sliderMoved, [=](int value) {
        QPoint pos = QCursor::pos();
        QToolTip::showText(pos, QString("%1").arg(value), nullptr);
    });
}

void ZenoParamSlider::setValue(int value) {
    m_pSlider->setValue(value);
}

void ZenoParamSlider::setSliderInfo(const SLIDER_INFO &info) {
    m_pSlider->setSingleStep(info.step);
    m_pSlider->setRange(info.min, info.max);
}

void ZenoParamSlider::updateStyleSheet() {
    m_pSlider->setStyleSheet(ZenoStyle::dpiScaleSheet(R"(
					QSlider {
					    background: rgb(31,39,42);
						margin-top : 8px;
					}
                    QSlider::groove:horizontal {
                        height: 4px;
                        background: #707D9C;
                    }
                    QSlider::handle:horizontal {
                        background: #DFE2E5;
                        width: 6px;
                        margin: -8px 0;
                    }
                    QSlider::add-page:horizontal {
                        background: #191D21;
                    }
                    
                    QSlider::sub-page:horizontal {
                        background: #707D9C;
                    }
    )"));
}

//////////////////////////////////////////////////////////////////////////////////////
ZenoParamSpinBoxSlider::ZenoParamSpinBoxSlider(Qt::Orientation orientation, int value, const SLIDER_INFO &info, QGraphicsItem *parent)
    : ZenoParamWidget(parent) 
{
    m_pSlider = new ZSpinBoxSlider();
    m_pSlider->setValue(value);
    m_pSlider->setSingleStep(info.step);
    m_pSlider->setRange(info.min, info.max);
    m_pSlider->setAttribute(Qt::WA_StyledBackground, false);
    setWidget(m_pSlider);
    updateStyleSheet();

    connect(m_pSlider, &ZSpinBoxSlider::valueChanged, this, &ZenoParamSpinBoxSlider::valueChanged);
}

void ZenoParamSpinBoxSlider::setValue(int value) {
    m_pSlider->setValue(value);
}

void ZenoParamSpinBoxSlider::setSliderInfo(const SLIDER_INFO &info) {
    m_pSlider->setSingleStep(info.step);
    m_pSlider->setRange(info.min, info.max);
}

void ZenoParamSpinBoxSlider::updateStyleSheet() {
    m_pSlider->setStyleSheet("background:rgb(31,39,42);");
}

//////////////////////////////////////////////////////////////////////////////////////
ZenoParamSpinBox::ZenoParamSpinBox(const SLIDER_INFO &info, QGraphicsItem *parent) : ZenoParamWidget(parent) 
{
    m_pSpinBox = new QSpinBox;
    m_pSpinBox->setProperty("cssClass", "control");
    m_pSpinBox->setAlignment(Qt::AlignCenter);
    m_pSpinBox->setFixedHeight(ZenoStyle::dpiScaled(24));
    m_pSpinBox->setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
    m_pSpinBox->setSingleStep(info.step);
    m_pSpinBox->setRange(info.min, info.max);
    setWidget(m_pSpinBox);
    connect(m_pSpinBox, SIGNAL(valueChanged(int)), this, SIGNAL(valueChanged(int)));
}

void ZenoParamSpinBox::setValue(int value) 
{
    m_pSpinBox->setValue(value);
}

void ZenoParamSpinBox::setSliderInfo(const SLIDER_INFO &info) 
{
    m_pSpinBox->setSingleStep(info.step);
    m_pSpinBox->setRange(info.min, info.max);
}

//////////////////////////////////////////////////////////////////////////////////////
ZenoParamDoubleSpinBox::ZenoParamDoubleSpinBox(const SLIDER_INFO &info, QGraphicsItem *parent)
{
    m_pSpinBox = new QDoubleSpinBox;
    m_pSpinBox->setAlignment(Qt::AlignCenter);
    m_pSpinBox->setFixedHeight(ZenoStyle::dpiScaled(24));
    m_pSpinBox->setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
    m_pSpinBox->setSingleStep(info.step);
    m_pSpinBox->setRange(info.min, info.max);
    setWidget(m_pSpinBox);
    connect(m_pSpinBox, SIGNAL(valueChanged(double)), this, SIGNAL(valueChanged(double)));
}

void ZenoParamDoubleSpinBox::setValue(double value) 
{
    m_pSpinBox->setValue(value);
}

void ZenoParamDoubleSpinBox::setSliderInfo(const SLIDER_INFO &info) 
{
    m_pSpinBox->setSingleStep(info.step);
    m_pSpinBox->setRange(info.min, info.max);
}

//////////////////////////////////////////////////////////////////////////////////////
ZenoParamGroupLine::ZenoParamGroupLine(const QString &text, QGraphicsItem *parent) : 
    QGraphicsItem(parent),
    m_text(text)
{

}

QRectF ZenoParamGroupLine::boundingRect() const 
{
    return QRectF(QPointF(0, 0), data(GVKEY_SIZEHINT).toSizeF());
}

void ZenoParamGroupLine::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) 
{
    //draw text
    QFont font = zenoApp->font();
    QFontMetrics fm(font);
    qreal width = fm.width(m_text);
    QPen pen;
    pen.setColor(QColor(94, 103, 115));
    painter->setPen(pen);
    painter->setFont(font);
    QRectF rect = this->boundingRect();
    qreal x = (rect.width() - width) / 2;
    rect.adjust(x, 0, -x, 0);
    painter->drawText(rect, Qt::AlignCenter, m_text);
    //draw line
    qreal y = rect.y() + (rect.height() / 2);
    pen.setColor(QColor(22, 25, 29));
    pen.setWidthF(ZenoStyle::dpiScaled(2));
    painter->setPen(pen);
    painter->drawLine(QPointF(0, y), QPointF(rect.left() - ZenoStyle::dpiScaled(4), y));
    painter->drawLine(QPointF(rect.right() + ZenoStyle::dpiScaled(4), y), QPointF(boundingRect().right(), y));
}

void ZenoParamGroupLine::setText(const QString &text) 
{
    m_text = text;
    update();
}

    //////////////////////////////////////////////////////////////////////////////////////
ZenoTextLayoutItem::ZenoTextLayoutItem(const QString &text, const QFont &font, const QColor &color, QGraphicsItem *parent)
    : QGraphicsLayoutItem()
    , QGraphicsTextItem(text, parent)
    , m_text(text)
    , m_bRight(false)
    , m_pSlider(nullptr)
{
    setZValue(ZVALUE_ELEMENT);
    setFont(font);
    setDefaultTextColor(color);
    
    setGraphicsItem(this);
    setFlags(ItemSendsScenePositionChanges);
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
}

void ZenoTextLayoutItem::setGeometry(const QRectF& geom)
{
    prepareGeometryChange();
    QGraphicsLayoutItem::setGeometry(geom);
    setPos(geom.topLeft());
    initAlignment(geom.width());
}

void ZenoTextLayoutItem::setRight(bool right)
{
    m_bRight = right;
}

void ZenoTextLayoutItem::initAlignment(qreal textWidth)
{
    if (m_bRight)
    {
        QTextDocument *doc = document();

        QTextCursor cursor = textCursor();
        cursor.movePosition(QTextCursor::Start);

        QTextFrame::iterator it;
        QTextFrame *rootFrame = doc->rootFrame();
        for (it = rootFrame->begin(); !(it.atEnd()); ++it)
        {
            QTextFrame *childFrame = it.currentFrame();
            QTextBlock childBlock = it.currentBlock();
            if (childBlock.isValid())
            {
                QTextBlockFormat format = childBlock.blockFormat();
                format.setAlignment(Qt::AlignRight);
                cursor.setBlockFormat(format);
            }
        }
        doc->setTextWidth(textWidth);
    }
}

void ZenoTextLayoutItem::setText(const QString& text)
{
    m_text = text;
    setPlainText(m_text);
}

void ZenoTextLayoutItem::setMargins(qreal leftM, qreal topM, qreal rightM, qreal bottomM)
{
    QTextFrame *frame = document()->rootFrame();
    QTextFrameFormat format = frame->frameFormat();
    format.setLeftMargin(leftM);
    format.setRightMargin(rightM);
    format.setTopMargin(topM);
    format.setBottomMargin(bottomM);
    frame->setFrameFormat(format);
}

void ZenoTextLayoutItem::setBackground(const QColor& clr)
{
    m_bg = clr;
}

QRectF ZenoTextLayoutItem::boundingRect() const
{
    QRectF rc = QRectF(QPointF(0, 0), geometry().size());
    return rc;
}

void ZenoTextLayoutItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    QStyleOptionGraphicsItem myOption(*option);
    myOption.state &= ~QStyle::State_Selected;
    myOption.state &= ~QStyle::State_HasFocus;
    if (m_bg.isValid())
    {
        painter->setPen(Qt::NoPen);
        painter->setBrush(m_bg);
        painter->drawRect(boundingRect());
    }
    QGraphicsTextItem::paint(painter, &myOption, widget);
}

QPainterPath ZenoTextLayoutItem::shape() const
{
    QPainterPath path;
    path.addRect(boundingRect());
    return path;
}

void ZenoTextLayoutItem::setScalesSlider(QGraphicsScene* pScene, const QVector<qreal>& scales)
{
    m_scales = scales;
    m_pSlider = new ZGraphicsNumSliderItem(scales, nullptr);
    if (pScene)
        pScene->addItem(m_pSlider);
    m_pSlider->setZValue(1000);
    m_pSlider->hide();
}

QSizeF ZenoTextLayoutItem::sizeHint(Qt::SizeHint which, const QSizeF &constraint) const
{
    QRectF rc = QGraphicsTextItem::boundingRect();
    switch (which)
    {
        case Qt::MinimumSize:
        case Qt::PreferredSize:
            return rc.size();
        case Qt::MaximumSize:
            return QSizeF(3000, rc.height());
        default:
            break;
    }
    return constraint;
}

void ZenoTextLayoutItem::focusOutEvent(QFocusEvent* event)
{
    QGraphicsTextItem::focusOutEvent(event);
    emit editingFinished();

    QString newText = document()->toPlainText();
    if (newText != m_text)
    {
        QString oldText = m_text;
        m_text = newText;
        emit contentsChanged(oldText, newText);
    }
}

void ZenoTextLayoutItem::hoverEnterEvent(QGraphicsSceneHoverEvent* event)
{
    QGraphicsTextItem::hoverEnterEvent(event);
    if (textInteractionFlags() & Qt::TextEditorInteraction)
        setCursor(QCursor(Qt::IBeamCursor));
}

void ZenoTextLayoutItem::hoverMoveEvent(QGraphicsSceneHoverEvent* event)
{
    QGraphicsTextItem::hoverMoveEvent(event);
}

void ZenoTextLayoutItem::hoverLeaveEvent(QGraphicsSceneHoverEvent* event)
{
    QGraphicsTextItem::hoverLeaveEvent(event);
    if (textInteractionFlags() & Qt::TextEditorInteraction)
        setCursor(QCursor(Qt::ArrowCursor));
}

void ZenoTextLayoutItem::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Shift)
    {
        if (m_pSlider)
        {
            m_pSlider->setPos(this->scenePos());
            m_pSlider->show();
        }
    }
    QGraphicsTextItem::keyPressEvent(event);
}

void ZenoTextLayoutItem::keyReleaseEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Shift)
    {
        if (m_pSlider)
        {
            m_pSlider->hide();
        }
    }
    QGraphicsTextItem::keyReleaseEvent(event);
}


//////////////////////////////////////////////////////////////////////////////////////
ZenoSpacerItem::ZenoSpacerItem(bool bHorizontal, qreal size, QGraphicsItem* parent)
    : QGraphicsLayoutItem()
    , QGraphicsItem(parent)
    , m_bHorizontal(bHorizontal)
    , m_size(size)
{
}

void ZenoSpacerItem::setGeometry(const QRectF& rect)
{
	prepareGeometryChange();
	QGraphicsLayoutItem::setGeometry(rect);
	setPos(rect.topLeft());
}

QRectF ZenoSpacerItem::boundingRect() const
{
    if (m_bHorizontal)
        return QRectF(0, 0, m_size, 0);
    else
        return QRectF(0, 0, 0, m_size);
}

void ZenoSpacerItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
}

QSizeF ZenoSpacerItem::sizeHint(Qt::SizeHint which, const QSizeF& constraint) const
{
	QRectF rc = boundingRect();
	switch (which)
	{
	case Qt::MinimumSize:
	case Qt::PreferredSize:
    case Qt::MaximumSize:
		return rc.size();
	default:
		break;
	}
	return constraint;
}

//////////////////////////////////////////////////////////////////////////////////////
ZenoBoardTextLayoutItem::ZenoBoardTextLayoutItem(const QString &text, const QFont &font, const QColor &color, const QSizeF& sz, QGraphicsItem *parent)
    : QGraphicsLayoutItem()
    , QGraphicsTextItem(text, parent)
    , m_text(text)
    , m_size(sz)
{
    setZValue(ZVALUE_ELEMENT);
    setFont(font);
    setDefaultTextColor(color);

    setGraphicsItem(this);
    setFlags(ItemIsFocusable | ItemIsSelectable | ItemSendsScenePositionChanges);
    setTextInteractionFlags(Qt::TextEditorInteraction);

    connect(document(), &QTextDocument::contentsChanged, this, [=]() {
        updateGeometry();
    });
}

void ZenoBoardTextLayoutItem::setGeometry(const QRectF& geom)
{
    prepareGeometryChange();
    QGraphicsLayoutItem::setGeometry(geom);
    setPos(geom.topLeft());
    //emit geometrySetup(scenePos());
}

QRectF ZenoBoardTextLayoutItem::boundingRect() const
{
    QRectF rc = QRectF(QPointF(0, 0), geometry().size());
    return rc;
}

void ZenoBoardTextLayoutItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    QGraphicsTextItem::paint(painter, option, widget);
}

QSizeF ZenoBoardTextLayoutItem::sizeHint(Qt::SizeHint which, const QSizeF& constraint) const
{
    return m_size;
    QRectF rc = QGraphicsTextItem::boundingRect();
    switch (which) {
        case Qt::MinimumSize:
        case Qt::PreferredSize:
            return rc.size();
        case Qt::MaximumSize:
            return QSizeF(1000, 1000);
        default:
            break;
    }
    return constraint;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
ZenoMinStatusBtnItem::ZenoMinStatusBtnItem(const StatusComponent& statusComp, QGraphicsItem* parent)
    : _base(parent)
    , m_minMute(nullptr)
    , m_minView(nullptr)
    , m_minOnce(nullptr)
{
    m_minMute = new ZenoImageItem(statusComp.mute, ZenoStyle::dpiScaledSize(QSize(48, 66)), this);
    m_minOnce = new ZenoImageItem(statusComp.once, ZenoStyle::dpiScaledSize(QSize(48, 66)), this);
    m_minView = new ZenoImageItem(statusComp.view, ZenoStyle::dpiScaledSize(QSize(37, 66)), this);
	m_once = new ZenoImageItem(
        ":/icons/ONCE_dark.svg",
        ":/icons/ONCE_light.svg",
        ":/icons/ONCE_light.svg",
        ZenoStyle::dpiScaledSize(QSize(50, 42)),
        this);
	m_mute = new ZenoImageItem(
        ":/icons/MUTE_dark.svg",
        ":/icons/MUTE_light.svg",
        ":/icons/MUTE_light.svg", 
        ZenoStyle::dpiScaledSize(QSize(50, 42)),
        this);
	m_view = new ZenoImageItem(
        ":/icons/VIEW_dark.svg",
        ":/icons/VIEW_light.svg",
        ":/icons/VIEW_light.svg",
        ZenoStyle::dpiScaledSize(QSize(50, 42)),
        this);

    //m_once->setFlag(QGraphicsItem::ItemIgnoresTransformations);
    //m_mute->setFlag(QGraphicsItem::ItemIgnoresTransformations);
    //m_view->setFlag(QGraphicsItem::ItemIgnoresTransformations);

    m_minMute->setCheckable(true);
    m_minView->setCheckable(true);
    m_minOnce->setCheckable(true);
    m_once->setCheckable(true);
    m_mute->setCheckable(true);
    m_view->setCheckable(true);
    m_once->hide();
    m_mute->hide();
    m_view->hide();

    m_minOnce->setPos(QPointF(0, 0));
    m_minMute->setPos(QPointF(ZenoStyle::dpiScaled(29), 0));
    m_minView->setPos(QPointF(ZenoStyle::dpiScaled(58), 0));

    QSizeF sz2 = m_once->size();
    qreal sMarginTwoBar = ZenoStyle::dpiScaled(4);
    //todo: kill these magin number.
    QPointF base = QPointF(ZenoStyle::dpiScaled(18), -sz2.height() - sMarginTwoBar);
	m_once->setPos(base);
	base += QPointF(ZenoStyle::dpiScaled(38), 0);
	m_mute->setPos(base);
	base += QPointF(ZenoStyle::dpiScaled(38), 0);
	m_view->setPos(base);

    onZoomed();

    m_minOnce->setZValue(ZVALUE_ELEMENT);
    m_minView->setZValue(ZVALUE_ELEMENT);
    m_minMute->setZValue(ZVALUE_ELEMENT);

    connect(m_minOnce, SIGNAL(hoverChanged(bool)), m_once, SLOT(setHovered(bool)));
    connect(m_minView, SIGNAL(hoverChanged(bool)), m_view, SLOT(setHovered(bool)));
    connect(m_minMute, SIGNAL(hoverChanged(bool)), m_mute, SLOT(setHovered(bool)));
	connect(m_once, SIGNAL(hoverChanged(bool)), m_minOnce, SLOT(setHovered(bool)));
	connect(m_view, SIGNAL(hoverChanged(bool)), m_minView, SLOT(setHovered(bool)));
	connect(m_mute, SIGNAL(hoverChanged(bool)), m_minMute, SLOT(setHovered(bool)));

	connect(m_minOnce, SIGNAL(toggled(bool)), m_once, SLOT(toggle(bool)));
	connect(m_minView, SIGNAL(toggled(bool)), m_view, SLOT(toggle(bool)));
	connect(m_minMute, SIGNAL(toggled(bool)), m_mute, SLOT(toggle(bool)));
	connect(m_once, SIGNAL(toggled(bool)), m_minOnce, SLOT(toggle(bool)));
	connect(m_view, SIGNAL(toggled(bool)), m_minView, SLOT(toggle(bool)));
	connect(m_mute, SIGNAL(toggled(bool)), m_minMute, SLOT(toggle(bool)));

    connect(m_minMute, &ZenoImageItem::toggled, [=](bool hovered) {
        emit toggleChanged(STATUS_MUTE, hovered);
    });
	connect(m_minView, &ZenoImageItem::toggled, [=](bool hovered) {
        emit toggleChanged(STATUS_VIEW, hovered);
    });
	connect(m_minOnce, &ZenoImageItem::toggled, [=](bool hovered) {
        emit toggleChanged(STATUS_ONCE, hovered);
	});

    setAcceptHoverEvents(true);
}

void ZenoMinStatusBtnItem::setOptions(int options)
{
    setChecked(STATUS_ONCE, options & OPT_ONCE);
    setChecked(STATUS_MUTE, options & OPT_MUTE);
    setChecked(STATUS_VIEW, options & OPT_VIEW);
}

void ZenoMinStatusBtnItem::setChecked(STATUS_BTN btn, bool bChecked)
{
    if (btn == STATUS_MUTE)
    {
        m_mute->toggle(bChecked);
        m_minMute->toggle(bChecked);
    }
    if (btn == STATUS_ONCE)
    {
		m_once->toggle(bChecked);
		m_minOnce->toggle(bChecked);
    }
	if (btn == STATUS_VIEW)
	{
		m_view->toggle(bChecked);
		m_minView->toggle(bChecked);
	}
}

void ZenoMinStatusBtnItem::hoverEnterEvent(QGraphicsSceneHoverEvent* event)
{
    m_mute->show();
    m_view->show();
    m_once->show();
    _base::hoverEnterEvent(event);
}

void ZenoMinStatusBtnItem::hoverMoveEvent(QGraphicsSceneHoverEvent* event)
{
    _base::hoverMoveEvent(event);
}

void ZenoMinStatusBtnItem::hoverLeaveEvent(QGraphicsSceneHoverEvent* event)
{
	m_mute->hide();
	m_view->hide();
	m_once->hide();
    _base::hoverLeaveEvent(event);
}

QRectF ZenoMinStatusBtnItem::boundingRect() const
{
    if (!m_mute->isVisible() && !m_view->isVisible() && !m_once->isVisible())
    {
        QRectF rc;
		rc = m_minMute->sceneBoundingRect();
		rc |= m_minView->sceneBoundingRect();
		rc |= m_minOnce->sceneBoundingRect();
        rc = mapRectFromScene(rc);
        return rc;
    }
    else
    {
		QRectF rc = childrenBoundingRect();
		return rc;
    }
}

void ZenoMinStatusBtnItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
}

void ZenoMinStatusBtnItem::onZoomed() 
{
    if (1 - editor_factor > 0.00001f) 
    {
        QSize size = ZenoStyle::dpiScaledSize(QSize(50 / editor_factor, 42 / editor_factor));
        m_once->resize(size);
        m_mute->resize(size);
        m_view->resize(size);
        QSizeF sz2 = m_once->size();
        qreal sMarginTwoBar = ZenoStyle::dpiScaled(4);
        QPointF base = QPointF(ZenoStyle::dpiScaled(18), -sz2.height() - sMarginTwoBar);
        m_once->setPos(base);
        base += QPointF(ZenoStyle::dpiScaled(38 / editor_factor), 0);
        m_mute->setPos(base);
        base += QPointF(ZenoStyle::dpiScaled(38 / editor_factor), 0);
        m_view->setPos(base);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
ZenoMinStatusBtnWidget::ZenoMinStatusBtnWidget(const StatusComponent& statusComp, QGraphicsItem* parent)
	: QGraphicsLayoutItem()
	, ZenoMinStatusBtnItem(statusComp, parent)
{
}

void ZenoMinStatusBtnWidget::updateGeometry()
{
    QGraphicsLayoutItem::updateGeometry();
}

void ZenoMinStatusBtnWidget::setGeometry(const QRectF& rect)
{
	prepareGeometryChange();
	QGraphicsLayoutItem::setGeometry(rect);
	setPos(rect.topLeft());
}

QRectF ZenoMinStatusBtnWidget::boundingRect() const
{
    return ZenoMinStatusBtnItem::boundingRect();
	QRectF rc = QRectF(QPointF(0, 0), geometry().size());
	return rc;
}

QSizeF ZenoMinStatusBtnWidget::sizeHint(Qt::SizeHint which, const QSizeF& constraint) const
{
	switch (which)
	{
	case Qt::MinimumSize:
	case Qt::PreferredSize:
	case Qt::MaximumSize:
    {
        QRectF rc = m_minMute->sceneBoundingRect();
        rc |= m_minOnce->sceneBoundingRect();
        rc |= m_minView->sceneBoundingRect();
        return rc.size();
    }
	default:
		break;
	}
	return constraint;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
ZenoSvgLayoutItem::ZenoSvgLayoutItem(const ImageElement &elem, const QSizeF &sz, QGraphicsItem *parent)
    : QGraphicsLayoutItem()
    , ZenoImageItem(elem, sz, parent)
{
    setGraphicsItem(this);
}

QSizeF ZenoSvgLayoutItem::sizeHint(Qt::SizeHint which, const QSizeF &constraint) const
{
    switch (which)
    {
        case Qt::MinimumSize:
        case Qt::PreferredSize:
        case Qt::MaximumSize:
            return ZenoImageItem::boundingRect().size();
        default:
            break;
    }
    return constraint;
}

QRectF ZenoSvgLayoutItem::boundingRect() const
{
    QRectF rc = QRectF(QPointF(0, 0), geometry().size());
    return rc;
}

void ZenoSvgLayoutItem::setGeometry(const QRectF &rect)
{
    prepareGeometryChange();
    QGraphicsLayoutItem::setGeometry(rect);
    setPos(rect.topLeft());
}

void ZenoSvgLayoutItem::updateGeometry()
{
    QGraphicsLayoutItem::updateGeometry();
}


/////////////////////////////////////////////////////////////////////////
SpacerLayoutItem::SpacerLayoutItem(QSizeF sz, bool bHorizontal, QGraphicsLayoutItem *parent, bool isLayout)
    : QGraphicsLayoutItem(parent, isLayout)
    , m_sz(sz)
{
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
}

QSizeF SpacerLayoutItem::sizeHint(Qt::SizeHint which, const QSizeF &constraint) const
{
    switch (which)
    {
        case Qt::MinimumSize:
        case Qt::PreferredSize:
            return m_sz;
        case Qt::MaximumSize:
            return m_sz;
        default:
            return m_sz;
    }
    return constraint;
}
