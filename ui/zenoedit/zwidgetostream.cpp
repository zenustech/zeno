#include "zwidgetostream.h"
#include "zenoapplication.h"
#include <zenoui/model/modelrole.h>


ZWidgetErrStream::ZWidgetErrStream()
    : std::basic_streambuf<char>()
    , m_stream(std::cerr)
{
    m_old_buf = m_stream.rdbuf();
    m_stream.rdbuf(this);
}

ZWidgetErrStream::~ZWidgetErrStream()
{
    m_stream.rdbuf(m_old_buf);
}

std::streamsize ZWidgetErrStream::xsputn(const char* p, std::streamsize n)
{
    QString str(p);

    //format like:
    //"[I 14:15:11.810] (unknown:0) begin frame 89"

    QRegExp rx("\\[(T|D|I|C|W|E)\\s+(\\d+):(\\d+):(\\d+)\\.(\\d+)\\]\\s+\\(([^\\)]+):(\\d+)\\)\\s+([^\\)]+)");
    int pos = rx.indexIn(str);
    QStringList list = rx.capturedTexts();
    int sz = list.length();
    if (sz == 9)
    {
        QString type = list[1];
        int nDays = list[2].toInt();
        int nHours = list[3].toInt();
        int nMins = list[4].toInt();
        int nSeconds = list[5].toInt();
        QString fileName = list[6];
        int line = list[7].toInt();
        QString content = list[8];
        QString msg = QString("[%1:%2:%3.%4] (%5:%6) %7").arg(
            QString::number(nDays), QString::number(nHours), QString::number(nMins), 
            QString::number(nSeconds), fileName, QString::number(line), content);

        QByteArray arr = fileName.toUtf8();
        const char *pwtf = arr.constData();
        QMessageLogger logger(pwtf, line, 0);
        if (type == "T")
        {
            logger.info() << msg;
        }
        else if (type == "D")
        {
            logger.debug() << msg;
        }
        else if (type == "I")
        {
            logger.info() << msg;
        }
        else if (type == "C")
        {
            logger.critical() << msg;
        }
        else if (type == "W")
        {
            logger.warning() << msg;
        }
        else if (type == "E")
        {
            logger.fatal(msg.toLatin1());
        }
    }
    return n;
}

void ZWidgetErrStream::registerMsgHandler()
{
    qInstallMessageHandler(customMsgHandler);
}

void ZWidgetErrStream::customMsgHandler(QtMsgType type, const QMessageLogContext& context, const QString& msg)
{
    QString fileName = QString::fromLatin1(context.file);
    int ln = context.line;
    QByteArray localMsg = msg.toLocal8Bit();
    
    QStandardItemModel* model = zenoApp->logModel();

    QStandardItem *item = new QStandardItem(msg);
    item->setData(type, ROLE_LOGTYPE);
    item->setData(fileName, ROLE_FILENAME);
    item->setData(ln, ROLE_LINENO);
    switch (type)
    {
        //todo: time
        case QtDebugMsg:
        {
            item->setData(QBrush(QColor(255, 255, 255, 0.7*255)), Qt::ForegroundRole);
            model->appendRow(item);
            break;
        }
        case QtCriticalMsg:
        {
            item->setData(QBrush(QColor(255, 255, 255, 0.7*255)), Qt::ForegroundRole);
            model->appendRow(item);
            break;
        }
        case QtInfoMsg:
        {
            item->setData(QBrush(QColor(51, 148, 85)), Qt::ForegroundRole);
            model->appendRow(item);
            break;
        }
        case QtWarningMsg:
        {
            item->setData(QBrush(QColor(200, 154, 80)), Qt::ForegroundRole);
            model->appendRow(item);
            break;
        }
        case QtFatalMsg:
        {
            item->setData(QBrush(QColor(200, 84, 79)), Qt::ForegroundRole);
            model->appendRow(item);
            break;
        }
    default:
        delete item;
        break;
    }
}