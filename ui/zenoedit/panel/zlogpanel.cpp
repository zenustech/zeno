#include "zlogpanel.h"
#include "ui_zlogpanel.h"
#include "zenoapplication.h"
#include <zenoui/model/modelrole.h>
#include <zenoui/util/uihelper.h>


LogItemDelegate::LogItemDelegate(QObject* parent)
    : _base(parent)
{

}

void LogItemDelegate::paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    painter->save();
    QtMsgType type = (QtMsgType)index.data(ROLE_LOGTYPE).toInt();
    QColor clr;
    if (type == QtFatalMsg) {
        clr = QColor(200, 84, 79);
    } else if (type == QtInfoMsg) {
        clr = QColor(51, 148, 85);
    } else if (type == QtWarningMsg) {
        clr = QColor(200, 154, 80);
    } else {
        clr = QColor("#858280");
    }
    QPen pen = painter->pen();
    pen.setColor(clr);
    painter->setPen(pen);
    _base::paint(painter, option, index);
    painter->restore();
}


LogListView::LogListView(QWidget* parent)
    : _base(parent)
{
}

void LogListView::rowsInserted(const QModelIndex& parent, int start, int end)
{
    _base::rowsInserted(parent, start, end);
    int n = model()->rowCount();
    //if (n > 0) {
    //    scrollTo(this->model()->index(n - 1, 0));
    //}
    //scrollToBottom();
}


ZlogPanel::ZlogPanel(QWidget* parent)
    : QWidget(parent)
    , m_pFilterModel(nullptr)
{
    m_ui = new Ui::LogPanel;
    m_ui->setupUi(this);

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

void ZlogPanel::initSignals()
{
    connect(m_ui->cbAll, &QCheckBox::stateChanged, this, [=](int state) {
        if (Qt::Checked == state) {
            BlockSignalScope s1(m_ui->cbCritical);
            BlockSignalScope s2(m_ui->cbDebug);
            BlockSignalScope s3(m_ui->cbError);
            BlockSignalScope s4(m_ui->cbInfo);
            BlockSignalScope s5(m_ui->cbWarning);
            m_ui->cbCritical->setChecked(true);
            m_ui->cbDebug->setChecked(true);
            m_ui->cbError->setChecked(true);
            m_ui->cbInfo->setChecked(true);
            m_ui->cbWarning->setChecked(true);
        } else if (Qt::Unchecked == state) {
        
        }
        onFilterChanged();
    });

    connect(m_ui->cbCritical, &QCheckBox::stateChanged, this, [=](int state) {
        if (Qt::Checked == state) {

        } else if (Qt::Unchecked == state) {
            BlockSignalScope scope(m_ui->cbAll);
            m_ui->cbAll->setChecked(false);
        }
        onFilterChanged();
    });

    connect(m_ui->cbDebug, &QCheckBox::stateChanged, this, [=](int state) {
        if (Qt::Checked == state) {

        } else if (Qt::Unchecked == state) {
            BlockSignalScope scope(m_ui->cbAll);
            m_ui->cbAll->setChecked(false);
        }
        onFilterChanged();
    });

    connect(m_ui->cbError, &QCheckBox::stateChanged, this, [=](int state) {
        if (Qt::Checked == state) {

        } else if (Qt::Unchecked == state) {
            BlockSignalScope scope(m_ui->cbAll);
            m_ui->cbAll->setChecked(false);
        }
        onFilterChanged();
    });

    connect(m_ui->cbInfo, &QCheckBox::stateChanged, this, [=](int state) {
        if (Qt::Checked == state) {

        } else if (Qt::Unchecked == state) {
            BlockSignalScope scope(m_ui->cbAll);
            m_ui->cbAll->setChecked(false);
        }
        onFilterChanged();
    });

    connect(m_ui->cbWarning, &QCheckBox::stateChanged, this, [=](int state) {
        if (Qt::Checked == state) {

        } else if (Qt::Unchecked == state) {
            BlockSignalScope scope(m_ui->cbAll);
            m_ui->cbAll->setChecked(false);
        }
        onFilterChanged();
    });
}

void ZlogPanel::onFilterChanged()
{
    QVector<QtMsgType> filters;
    if (m_ui->cbWarning->isChecked())
        filters.append(QtWarningMsg);
    if (m_ui->cbCritical->isChecked())
        filters.append(QtCriticalMsg);
    if (m_ui->cbDebug->isChecked())
        filters.append(QtDebugMsg);
    if (m_ui->cbError->isChecked())
        filters.append(QtFatalMsg);
    if (m_ui->cbInfo->isChecked())
        filters.append(QtInfoMsg);
    m_pFilterModel->setFilters(filters);
}


//////////////////////////////////////////////////////////////////////////
CustomFilterProxyModel::CustomFilterProxyModel(QObject *parent)
    : QSortFilterProxyModel(parent)
    , m_filters(0)
    
{
}

void CustomFilterProxyModel::setFilters(const QVector<QtMsgType>& filters)
{
    m_filters = filters;
    invalidate();
}

bool CustomFilterProxyModel::filterAcceptsRow(int source_row, const QModelIndex &source_parent) const
{
    QModelIndex index = sourceModel()->index(source_row, 0, source_parent);
    int role = filterRole();
    QtMsgType type = (QtMsgType)index.data(ROLE_LOGTYPE).toInt();
    if (m_filters.contains(type))
        return true;
    return false;
}