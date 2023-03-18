#include "ui_zlogpanel.h"
#include "zlogpanel.h"
#include "zenoapplication.h"
#include <zenomodel/include/igraphsmodel.h>
#include <zenomodel/include/modelrole.h>
#include <zenomodel/include/uihelper.h>
#include <zenomodel/include/graphsmanagment.h>
#include <zenoui/style/zenostyle.h>
#include <zenoui/comctrl/ztoolbutton.h>
#include "zenomainwindow.h"
#include "nodesview/zenographseditor.h"
#include "settings/zenosettingsmanager.h"


LogItemDelegate::LogItemDelegate(QObject* parent)
    : _base(parent)
{
    m_view = qobject_cast<QAbstractItemView*>(parent);

    m_textMargins.setLeft(ZenoStyle::dpiScaled(4));
    m_textMargins.setRight(ZenoStyle::dpiScaled(4));
    m_textMargins.setTop(ZenoStyle::dpiScaled(2));
    m_textMargins.setBottom(ZenoStyle::dpiScaled(4));
}

QSize LogItemDelegate::sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    QFont font = getFont();
    QFontMetrics fm(font);
    const QAbstractItemModel* model = index.model();
    QString Text = model->data(index, Qt::DisplayRole).toString();
    QRect neededsize = fm.boundingRect(option.rect, Qt::TextWordWrap, Text);

    QVector<QTextLayout::FormatRange> selections;
    QTextLayout textLayout;
    initTextLayout(Text, font, m_view->width(), textLayout, selections);

    qreal height = textLayout.boundingRect().height();

    return QSize(option.rect.width() + m_textMargins.left() + m_textMargins.right(), 
                 height + m_textMargins.top() + m_textMargins.bottom());
}

void LogItemDelegate::initStyleOption(QStyleOptionViewItem* option,
    const QModelIndex& index) const
{
    QStyledItemDelegate::initStyleOption(option, index);
    QFont font = getFont();
    option->font = font;
}

QFont LogItemDelegate::getFont() const
{
    QFont font = zenoApp->font();
    font.setWeight(QFont::DemiBold);
    return font;
}

void LogItemDelegate::paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    painter->save();

    QStyleOptionViewItem opt = option;
    initStyleOption(&opt, index);

    QtMsgType type = (QtMsgType)index.data(ROLE_LOGTYPE).toInt();
    QColor clr;
    if (type == QtFatalMsg) {
        clr = QColor("#C8544F");
    } else if (type == QtInfoMsg) {
        clr = QColor("#507CC8");
    } else if (type == QtWarningMsg) {
        clr = QColor("#C89A50");
    }
    else if (type == QtCriticalMsg) {
        clr = QColor("#339455");
    }
    else if (type == QtDebugMsg) {
        clr = QColor("#A3B1C0");
    }
    else {
        clr = QColor("#A3B1C0");
    }

    QRect rc = opt.rect;
    if (opt.state & QStyle::State_Selected)
    {
        painter->fillRect(rc, QColor("#3B546D"));
    }
    else if (opt.state & QStyle::State_MouseOver)
    {
        painter->fillRect(rc, QColor("#24282E"));
    }

    QPen pen = painter->pen();
    pen.setColor(clr);

    QFont font = getFont();
    painter->setFont(font);
    painter->setPen(pen);

    QRect textRect = rc.adjusted(m_textMargins.left(), m_textMargins.top(), 0, 0);
    QPointF paintPosition = textRect.topLeft();

    QVector<QTextLayout::FormatRange> selections;
    QTextLayout textLayout;
    initTextLayout(opt.text, font, rc.width(), textLayout, selections);

    selections.clear();

    int rgStart = index.data(ROLE_RANGE_START).toInt();
    int rgLen = index.data(ROLE_RANGE_LEN).toInt();
    if (rgLen > 0 && (opt.state & QStyle::State_MouseOver))
    {
        QTextLayout::FormatRange rg;
        rg.start = rgStart;
        rg.length = rgLen;
        rg.format.setFontUnderline(true);
        selections.append(rg);
    }

    textLayout.draw(painter, paintPosition, selections);

    painter->setPen(QColor("#24282E"));
    painter->drawLine(rc.bottomLeft(), rc.bottomRight());

    painter->restore();
}

void LogItemDelegate::initTextLayout(
                    const QString& text,
                    const QFont& font,
                    qreal fixedWidth,
                    QTextLayout& textLayout,
                    QVector<QTextLayout::FormatRange>& selections) const
{
    QTextOption textOption;
    textOption.setWrapMode(QTextOption::WrapAnywhere);
    textOption.setTextDirection(Qt::LeftToRight);

    textLayout.setText(text);
    textLayout.setFont(font);
    textLayout.setTextOption(textOption);
    UiHelper::viewItemTextLayout(textLayout, fixedWidth);

    selections = _getNodeIdentRgs(text);
}

QVector<QTextLayout::FormatRange> LogItemDelegate::_getNodeIdentRgs(const QString& content) const
{
    QVector<QTextLayout::FormatRange> selections;
    QRegExp rx("[0-9a-z]+\\-[^/\\s`'\"\\]]+");
    QString currText = content;

    int index = -1;
    while ((index = rx.indexIn(currText, index + 1)) >= 0) {
        int capLen = rx.cap(0).length();
        QString strIdent = currText.mid(index, capLen);

        QTextLayout::FormatRange frg;
        frg.start = index;
        frg.length = strIdent.length();
        frg.format.setFontUnderline(true);

        selections.push_back(frg);
        index += strIdent.length();
    }
    return selections;
}

QTextLayout::FormatRange LogItemDelegate::getHoverRange(const QString& text, qreal mouseX, qreal mouseY, QRect rc) const
{
    QVector<QTextLayout::FormatRange> selections = _getNodeIdentRgs(text);
    if (!selections.isEmpty())
    {
        QFont font = getFont();

        QVector<QTextLayout::FormatRange> selections;
        QTextLayout textLayout;
        initTextLayout(text, font, rc.width(), textLayout, selections);

        //calculate row and column of mousePos.
        qreal h = 0;
        int r = 0;
        for (; r < textLayout.lineCount(); r++)
        {
            QTextLine line = textLayout.lineAt(r);
            h += line.height();
            if (mouseY < h) {
                //in row r.
                int cursor = line.xToCursor(mouseX);
                for (QTextLayout::FormatRange rg : selections)
                {
                    if (rg.start <= cursor && cursor <= rg.start + rg.length)
                    {
                        return rg;
                    }
                }
            }
        }
    }
    return QTextLayout::FormatRange();
}

QWidget* LogItemDelegate::createEditor(QWidget* parent, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    return _base::createEditor(parent, option, index);
}

bool LogItemDelegate::editorEvent(QEvent* event, QAbstractItemModel* model, const QStyleOptionViewItem& option, const QModelIndex& index)
{
    if (event->type() == QEvent::MouseMove ||
        event->type() == QEvent::MouseButtonRelease)
    {
        QMouseEvent *pEvent = static_cast<QMouseEvent *>(event);
        QPoint mousePos = pEvent->pos();
        QString text = index.data().toString();
        qreal mouseX = mousePos.x() - m_textMargins.left();
        qreal mouseY = mousePos.y() - option.rect.top() - m_textMargins.top();
        QTextLayout::FormatRange rg = getHoverRange(text, mouseX, mouseY, option.rect);

        ZASSERT_EXIT(m_view, false);

        QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(index.model());
        ZASSERT_EXIT(pModel, false);
        pModel->setData(index, rg.start, ROLE_RANGE_START);
        pModel->setData(index, rg.length, ROLE_RANGE_LEN);

        if (event->type() == QEvent::MouseMove)
        {
            if (rg.length > 0) {
                m_view->setCursor(Qt::PointingHandCursor);
            } else {
                m_view->setCursor(Qt::ArrowCursor);
            }
        }
        else {
            if (rg.length > 0) {
                QString ident = text.mid(rg.start, rg.length);
                auto graphsMgm = zenoApp->graphsManagment();
                ZASSERT_EXIT(graphsMgm, false);
                IGraphsModel* pModel = graphsMgm->currentModel();
                ZASSERT_EXIT(pModel, false);
                QModelIndex idx = pModel->nodeIndex(ident);
                if (idx.isValid())
                {
                    QModelIndex subgIdx = idx.data(ROLE_SUBGRAPH_IDX).toModelIndex();
                    const QString& subgName = subgIdx.data(ROLE_OBJNAME).toString();
                    ZenoMainWindow* pWin = zenoApp->getMainWindow();
                    ZASSERT_EXIT(pWin, false);
                    ZenoGraphsEditor* pEditor = pWin->getAnyEditor();
                    if (pEditor) {
                        pEditor->activateTab(subgName, "", ident, false);
                    }
                }
            }
        }
        m_view->update(index);
    }
    return QStyledItemDelegate::editorEvent(event, model, option, index);
}


////////////////////////////////////////////////////////////////
LogListView::LogListView(QWidget* parent)
    : _base(parent)
{
    setItemDelegate(new LogItemDelegate(this));
    setWordWrap(true);
    setMouseTracking(true);
    setContextMenuPolicy(Qt::CustomContextMenu);
    connect(this, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(onCustomContextMenu(const QPoint&)));
}

void LogListView::rowsInserted(const QModelIndex& parent, int start, int end)
{
    _base::rowsInserted(parent, start, end);

    connect(&m_timer, &QTimer::timeout, this, [=]() {
        scrollToBottom();
        m_timer.stop();
    });
    m_timer.start(50);
}

void LogListView::onCustomContextMenu(const QPoint& point)
{
    QModelIndex index = indexAt(point);
    QString msg = index.data().toString();
    QMenu* pMenu = new QMenu;
    pMenu->setAttribute(Qt::WA_DeleteOnClose);

    QAction* pCopy = new QAction(tr("Copy"));
    pMenu->addAction(pCopy);
    connect(pCopy, &QAction::triggered, [=]() {
        QMimeData* pMimeData = new QMimeData;
        pMimeData->setText(msg);
        QApplication::clipboard()->setMimeData(pMimeData);
    });

    pMenu->exec(QCursor::pos());
}


ZPlainLogPanel::ZPlainLogPanel(QWidget* parent)
    : QPlainTextEdit(parent)
{
    setProperty("cssClass", "logpanel");
    setReadOnly(true);
    connect(zenoApp->logModel(), &QStandardItemModel::rowsInserted, this, [=](const QModelIndex& parent, int first, int last) {
        QStandardItemModel* pModel = qobject_cast<QStandardItemModel*>(sender());
        if (pModel) {
            QModelIndex idx = pModel->index(first, 0, parent);
            QString content = idx.data().toString();
            appendPlainText(content);
            verticalScrollBar()->setValue(verticalScrollBar()->maximum());
        }
    });
}


ZlogPanel::ZlogPanel(QWidget* parent)
    : QWidget(parent)
    , m_pFilterModel(nullptr)
    , m_pMenu(nullptr)
{
    m_ui = new Ui::LogPanel;
    m_ui->setupUi(this);

    m_ui->btnDebug->setButtonOptions(ZToolButton::Opt_Checkable | ZToolButton::Opt_HasIcon | ZToolButton::Opt_NoBackground);
    m_ui->btnDebug->setIcon(ZenoStyle::dpiScaledSize(QSize(20, 20)),
        ":/icons/logger_debug_unchecked.svg",
        ":/icons/logger_debug_unchecked.svg",
        ":/icons/logger_debug_checked.svg",
        ":/icons/logger_debug_checked.svg");
    m_ui->btnDebug->setChecked(true);

    m_ui->btnInfo->setButtonOptions(ZToolButton::Opt_Checkable | ZToolButton::Opt_HasIcon | ZToolButton::Opt_NoBackground);
    m_ui->btnInfo->setIcon(ZenoStyle::dpiScaledSize(QSize(20, 20)),
        ":/icons/logger_info_unchecked.svg",
        ":/icons/logger_info_unchecked.svg",
        ":/icons/logger_info_checked.svg",
        ":/icons/logger_info_checked.svg");
    m_ui->btnInfo->setChecked(true);

    m_ui->btnWarn->setButtonOptions(ZToolButton::Opt_Checkable | ZToolButton::Opt_HasIcon | ZToolButton::Opt_NoBackground);
    m_ui->btnWarn->setIcon(ZenoStyle::dpiScaledSize(QSize(20, 20)),
        ":/icons/logger_warning_unchecked.svg",
        ":/icons/logger_warning_unchecked.svg",
        ":/icons/logger_warning_checked.svg",
        ":/icons/logger_warning_checked.svg");
    m_ui->btnWarn->setChecked(true);

    m_ui->btnError->setButtonOptions(ZToolButton::Opt_Checkable | ZToolButton::Opt_HasIcon | ZToolButton::Opt_NoBackground);
    m_ui->btnError->setIcon(ZenoStyle::dpiScaledSize(QSize(20, 20)),
        ":/icons/logger_error_unchecked.svg",
        ":/icons/logger_error_unchecked.svg",
        ":/icons/logger_error_checked.svg",
        ":/icons/logger_error_checked.svg");
    m_ui->btnError->setChecked(true);

    m_ui->btnKey->setButtonOptions(ZToolButton::Opt_Checkable | ZToolButton::Opt_HasIcon | ZToolButton::Opt_NoBackground);
    m_ui->btnKey->setIcon(ZenoStyle::dpiScaledSize(QSize(20, 20)),
        ":/icons/logger-key-unchecked.svg",
        ":/icons/logger-key-unchecked.svg",
        ":/icons/logger-key-checked.svg",
        ":/icons/logger-key-checked.svg");
    m_ui->btnKey->setChecked(true);

    m_ui->editSearch->setProperty("cssClass", "zeno2_2_lineedit");
    m_ui->editSearch->setPlaceholderText(tr("Search"));

    m_ui->btnDelete->setButtonOptions(ZToolButton::Opt_HasIcon | ZToolButton::Opt_NoBackground);
    m_ui->btnDelete->setIcon(ZenoStyle::dpiScaledSize(QSize(20, 20)),
        ":/icons/toolbar_delete_idle.svg",
        ":/icons/toolbar_delete_light.svg",
        "",
        "");

    m_ui->btnSetting->setButtonOptions(ZToolButton::Opt_HasIcon | ZToolButton::Opt_NoBackground);
    m_ui->btnSetting->setIcon(ZenoStyle::dpiScaledSize(QSize(20, 20)),
        ":/icons/settings.svg",
        ":/icons/settings-on.svg",
        "",
        "");

    initSignals();
    initModel();
    onFilterChanged();
}

void ZlogPanel::initModel()
{
    m_pFilterModel = new CustomFilterProxyModel(this);
    m_pFilterModel->setSourceModel(zenoApp->logModel());
    m_pFilterModel->setFilterRole(ROLE_LOGTYPE);
    m_ui->listView->setModel(m_pFilterModel);
}

void ZlogPanel::onSettings()
{
    if (!m_pMenu)
    {
        m_pMenu = new QMenu(this);

        QAction *pFocusError = new QAction(tr("Trace Error"));
        pFocusError->setCheckable(true);
        m_pMenu->addAction(pFocusError);
        connect(pFocusError, &QAction::toggled, [=](bool bChecked) {
            ZenoSettingsManager::GetInstance().setValue(zsTraceErrorNode, bChecked);
        });
    }
    m_pMenu->exec(QCursor::pos());
}

void ZlogPanel::initSignals()
{
    connect(m_ui->btnKey, &ZToolButton::toggled, this, [=](bool bOn) {
        onFilterChanged();
    });

    connect(m_ui->btnDebug, &ZToolButton::toggled, this, [=](bool bOn) {
        onFilterChanged();
    });

    connect(m_ui->btnError, &ZToolButton::toggled, this, [=](bool bOn) {
        onFilterChanged();
    });

    connect(m_ui->btnInfo, &ZToolButton::toggled, this, [=](bool bOn) {
        onFilterChanged();
    });

    connect(m_ui->btnWarn, &ZToolButton::toggled, this, [=](bool bOn) {
        onFilterChanged();
    });

    connect(m_ui->editSearch, &QLineEdit::textChanged, this, [=](const QString& wtf) {
        onFilterChanged();
    });

    connect(m_ui->btnDelete, &ZToolButton::clicked, this, [=]() {
        zenoApp->logModel()->clear();
    });

    connect(m_ui->btnSetting, &ZToolButton::clicked, this, [=]() {
        onSettings();
    });
}

void ZlogPanel::onFilterChanged()
{
    QVector<QtMsgType> filters;
    if (m_ui->btnWarn->isChecked())
        filters.append(QtWarningMsg);
    if (m_ui->btnKey->isChecked())
        filters.append(QtCriticalMsg);
    if (m_ui->btnDebug->isChecked())
        filters.append(QtDebugMsg);
    if (m_ui->btnError->isChecked())
        filters.append(QtFatalMsg);
    if (m_ui->btnInfo->isChecked())
        filters.append(QtInfoMsg);
    m_pFilterModel->setFilters(filters, m_ui->editSearch->text());
}


//////////////////////////////////////////////////////////////////////////
CustomFilterProxyModel::CustomFilterProxyModel(QObject *parent)
    : QSortFilterProxyModel(parent)
    , m_filters(0)
{
}

void CustomFilterProxyModel::setFilters(const QVector<QtMsgType>& filters, const QString& content)
{
    if (m_filters != filters || m_searchContent != content)
    {
        m_filters = filters;
        m_searchContent = content;
        invalidate();
    }
}

bool CustomFilterProxyModel::filterAcceptsRow(int source_row, const QModelIndex &source_parent) const
{
    QModelIndex index = sourceModel()->index(source_row, 0, source_parent);
    int role = filterRole();
    QtMsgType type = (QtMsgType)index.data(ROLE_LOGTYPE).toInt();
    QString msg = index.data(Qt::DisplayRole).toString();
    if (m_filters.contains(type))
    {
        if (!m_searchContent.isEmpty())
        {
            if (msg.contains(m_searchContent, Qt::CaseInsensitive))
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        return true;
    }
    return false;
}