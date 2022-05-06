#include "zwidgetostream.h"
#include "zenoapplication.h"
#include <zenoui/model/modelrole.h>
#include "graphsmanagment.h"


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
        QString msg = QString("[%1:%2:%3.%4] (%5:%6) %7")
            .arg(nDays, 2, 10, QLatin1Char('0'))
            .arg(nHours, 2, 10, QLatin1Char('0'))
            .arg(nMins, 2, 10, QLatin1Char('0'))
            .arg(nSeconds, 3, 10, QLatin1Char('0'))
            .arg(fileName)
            .arg(line)
            .arg(content);

        QByteArray arr = fileName.toUtf8();
        const char *pwtf = arr.constData();
        QMessageLogger logger(pwtf, line, 0);
        if (type == "T")
        {
            logger.info().noquote() << msg;
        }
        else if (type == "D")
        {
            logger.debug().noquote() << msg;
        }
        else if (type == "I")
        {
            logger.info().noquote() << msg;
        }
        else if (type == "C")
        {
            logger.critical().noquote() << msg;
        }
        else if (type == "W")
        {
            logger.warning().noquote() << msg;
        }
        else if (type == "E")
        {
            logger.info().noquote() << "[E]" << msg;
            //crash when use logger.fatal.
            //logger.fatal(msg.toLatin1());
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
    if (msg.startsWith("[E]"))
    {
        type = QtFatalMsg;
    }
    auto gm = zenoApp->graphsManagment();
    gm->appendLog(type, fileName, context.line, msg);
}