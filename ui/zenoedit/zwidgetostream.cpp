#include "zwidgetostream.h"

myConsoleStream::myConsoleStream(std::ostream &stream, QTextEdit *text_edit)
    : std::basic_streambuf<char>(), m_stream(stream)

{
    this->log_window = text_edit;
    this->m_old_buf = stream.rdbuf();
    stream.rdbuf(this);
}

myConsoleStream::~myConsoleStream()
{
    this->m_stream.rdbuf(this->m_old_buf);
}

void myConsoleStream::registerMyConsoleMessageHandler()
{
    qInstallMessageHandler(myConsoleMessageHandler);
}

void myConsoleStream::myConsoleMessageHandler(QtMsgType type, const QMessageLogContext &, const QString &msg)
{
    QByteArray localMsg = msg.toLocal8Bit();
    switch (type)
    {
    case QtDebugMsg:
        // fprintf(stderr, "Debug: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        break;
    case QtInfoMsg:
        // fprintf(stderr, "Info: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        break;
    case QtWarningMsg:
        // fprintf(stderr, "Warning: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        break;
    case QtCriticalMsg:
        //fprintf(stderr, "Critical: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        break;
    case QtFatalMsg:
        // fprintf(stderr, "Fatal: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        break;
    default:
        std::cout << msg.toStdString().c_str();
        break;
    }
}


ZWidgetOutStream::ZWidgetOutStream()
    : std::basic_streambuf<char>()
    , m_stream(std::cout)
{
    m_old_buf = m_stream.rdbuf();
    m_stream.rdbuf(this);
}

ZWidgetOutStream::~ZWidgetOutStream()
{
    m_stream.rdbuf(m_old_buf);
}

std::streamsize ZWidgetOutStream::xsputn(const char* p, std::streamsize n)
{
    QString str(p);
    qDebug() << str;
    return n;
}

void ZWidgetOutStream::registerMsgHandler()
{
    qInstallMessageHandler(customMsgHandler);
}

void ZWidgetOutStream::customMsgHandler(QtMsgType type, const QMessageLogContext& context, const QString& msg)
{
    QByteArray localMsg = msg.toLocal8Bit();
    switch (type)
    {
    case QtDebugMsg:
        // fprintf(stderr, "Debug: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        break;
    case QtInfoMsg:
        // fprintf(stderr, "Info: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        break;
    case QtWarningMsg:
        // fprintf(stderr, "Warning: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        break;
    case QtCriticalMsg:
        // fprintf(stderr, "Critical: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        break;
    case QtFatalMsg:
        // fprintf(stderr, "Fatal: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        break;
    default:
        std::cout << msg.toStdString().c_str(); break;
    }
}

/////////////////////////////////////////////////////////////
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
            logger.info() << content;
        }
        else if (type == "D")
        {
            logger.debug() << content;
        }
        else if (type == "I")
        {
            logger.info() << content;
        }
        else if (type == "C")
        {
            logger.critical() << content;
        }
        else if (type == "W")
        {
            logger.warning() << content;
        }
        else if (type == "E")
        {
            logger.fatal(content.toLatin1());
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
    QByteArray localMsg = msg.toLocal8Bit();
    switch (type) {
    case QtDebugMsg:
        // fprintf(stderr, "Debug: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        break;
    case QtInfoMsg:
        // fprintf(stderr, "Info: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        break;
    case QtWarningMsg:
        // fprintf(stderr, "Warning: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        break;
    case QtCriticalMsg:
        // fprintf(stderr, "Critical: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        break;
    case QtFatalMsg:
        // fprintf(stderr, "Fatal: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        break;
    default:
        std::cout << msg.toStdString().c_str();
        break;
    }
    std::cout << msg.toStdString().c_str();
}