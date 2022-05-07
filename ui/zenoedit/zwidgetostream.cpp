#include "zwidgetostream.h"
#include "zenoapplication.h"
#include <zenoui/model/modelrole.h>
#include "graphsmanagment.h"


ZWidgetErrStream::ZWidgetErrStream(std::ostream &stream)
    : std::basic_streambuf<char>()
    , m_stream(stream)
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
    for (auto q = p; q != p + n; ++q) // make it visible to both real-console and luzh-log-panel
        putchar(*q);
    if (auto it = std::find(p, p + n, '\n'); it == p + n) {
        m_linebuffer.append(p, n);
    } else {
        m_linebuffer.append(p, it);
        //if (m_linebuffer.size() > 4 && std::equal(m_linebuffer.end() - 4, m_linebuffer.end(), "\033[0m")) {
            //m_linebuffer.erase(m_linebuffer.size() - 4);
        //}
        luzhPutString(QString::fromStdString(m_linebuffer));
        m_linebuffer.assign(it + 1, p + n - (it + 1));
    }
    return n;
}

void ZWidgetErrStream::luzhPutString(QString str) {
    //format like:
    //"[I 14:15:11.810] (unknown:0) begin frame 89"

    static QRegExp rx("\\[(T|D|I|C|W|E)\\s+(\\d+):(\\d+):(\\d+)\\.(\\d+)\\]\\s+\\(([^\\)]+):(\\d+)\\)\\s+([^\\)]+)");
    if (!str.startsWith('[') || rx.indexIn(str) == -1)
    {
        QMessageLogger logger("<stderr>", 0, 0);
        logger.critical().noquote() << "[C] <stderr>" << str;
    }
    else if (QStringList list = rx.capturedTexts(); list.length() == 9)
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
            logger.debug().noquote() << "[T]" << msg;
        }
        else if (type == "D")
        {
            logger.debug().noquote() << "[D]" << msg;
        }
        else if (type == "I")
        {
            logger.info().noquote() << "[I]" << msg;
        }
        else if (type == "C")
        {
            logger.critical().noquote() << "[C]" << msg;
        }
        else if (type == "W")
        {
            logger.warning().noquote() << "[W]" << msg;
        }
        else if (type == "E")
        {
            logger.warning().noquote() << "[E]" << msg;
            //crash when use logger.fatal.
            //logger.fatal(msg.toLatin1());
        }
    }
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
