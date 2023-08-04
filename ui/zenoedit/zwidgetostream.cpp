#include "zwidgetostream.h"
#include "zenoapplication.h"
#include <zenomodel/include/modelrole.h>
#include <zenomodel/include/graphsmanagment.h>
#include <cstring>


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

bool ZWidgetErrStream::isGUIThread()
{
    //return true;
    return QThread::currentThread() == qApp->thread();
}

std::streamsize ZWidgetErrStream::xsputn(const char* p, std::streamsize n)
{
    if (!isGUIThread()) {
        return _base::xsputn(p, n);
    }

    for (auto q = p; q != p + n; ++q) // make it visible to both real-console and luzh-log-panel
        putchar(*q);
    if (auto it = std::find(p, p + n, '\n'); it == p + n) {
        m_linebuffer.append(p, n);
    } else {
        m_linebuffer.append(p, it);
        //if (m_linebuffer.size() >= 4 && m_linebuffer.front() == '\033' && m_linebuffer[1] == '[') {
            //if (auto it = std::find(m_linebuffer.begin(), m_linebuffer.end(), 'm'); it != m_linebuffer.end())
                //m_linebuffer.erase(m_linebuffer.begin(), it + 1);
        //}
        //if (m_linebuffer.size() >= 4 && std::equal(m_linebuffer.end() - 4, m_linebuffer.end(), "\033[0m")) {
            //m_linebuffer.erase(m_linebuffer.size() - 4);
        //}
        appendFormatMsg(m_linebuffer);
        m_linebuffer.assign(it + 1, p + n - (it + 1));
    }
    return n;
}

void ZWidgetErrStream::appendFormatMsg(std::string const &str) {
    //format like:
    //"[I 14:15:11.810] (unknown:0) begin frame 89"
    if (!isGUIThread()) {
        return;
    }

    QMessageLogger logger("zeno", 0, 0);

    char type = 'C';
    // "[T "
    if (str.size() > 2 && str[0] == '[' && str[2] == ' ' && std::strchr("TDICWE", str[1])) {
        // "[T HH:MM:SS.sss] (file.cpp:42)"
        auto pos = str.find(')');
        if (pos != std::string::npos) {
            type = str[1];
        }
    }
    QString msg = QString::fromStdString(str);

    if (type == 'T')
    {
        logger.debug().noquote() << msg;
    }
    else if (type == 'D')
    {
        logger.debug().noquote() << msg;
    }
    else if (type == 'I')
    {
        logger.info().noquote() << msg;
    }
    else if (type == 'C')
    {
        logger.critical().noquote() << msg;
    }
    else if (type == 'W')
    {
        logger.warning().noquote() << msg;
    }
    else if (type == 'E')
    {
        logger.warning().noquote() << msg;
        //crash when use logger.fatal.
        //logger.fatal(msg.toUtf8());
    }
}

void ZWidgetErrStream::registerMsgHandler()
{
    qInstallMessageHandler(customMsgHandler);
}

void ZWidgetErrStream::customMsgHandler(QtMsgType type, const QMessageLogContext& context, const QString& msg)
{
    if (!isGUIThread()) {
        return;
    }

    if (!zenoApp) return;
    QString fileName = QString::fromUtf8(context.file);
    int ln = context.line;
    if (msg.startsWith("[E "))
    {
        type = QtFatalMsg;
    }
    auto gm = zenoApp->graphsManagment();
    gm->appendLog(type, fileName, context.line, msg);
}
