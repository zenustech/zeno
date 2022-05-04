#include "zlogpanel.h"
#include "ui_zlogpanel.h"
#include "zenoapplication.h"
#include <zenoui/model/modelrole.h>


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


ZlogPanel::ZlogPanel(QWidget* parent)
    : QWidget(parent)
{
    m_ui = new Ui::LogPanel;
    m_ui->setupUi(this);

    m_ui->listView->setModel(zenoApp->logModel());
    //m_ui->listView->setItemDelegate(new LogItemDelegate(m_ui->listView));
}