#ifndef __ZWIDGET_OSTREAM_H__
#define __ZWIDGET_OSTREAM_H__

#include <iostream>
#include <QtWidgets>

class myConsoleStream : public std::basic_streambuf<char>
{
  public:
    myConsoleStream(std::ostream &stream, QTextEdit *text_edit);

    virtual ~myConsoleStream();
    static void registerMyConsoleMessageHandler();

  private:
    static void myConsoleMessageHandler(QtMsgType type, const QMessageLogContext &, const QString &msg);

  protected:
    //This is called when a std::endl has been inserted into the stream
    virtual int_type overflow(int_type v) {
        if (v == '\n' && log_window) {
            log_window->append("");
        }
        return v;
    }

    virtual std::streamsize xsputn(const char *p, std::streamsize n) {
        //can do something.
        return std::basic_streambuf<char>::xsputn(p, n);
    }

  private:
    std::ostream &m_stream;
    std::streambuf *m_old_buf;
    QTextEdit *log_window;
};

class ZWidgetOutStream : public std::basic_streambuf<char>
{
    typedef std::basic_streambuf<char> _base;
public:
    ZWidgetOutStream();
    virtual ~ZWidgetOutStream();
    static void registerMsgHandler();

protected:
    virtual std::streamsize xsputn(const char* p, std::streamsize n);
    virtual int_type overflow(int_type v)
    {
        return v;
    }

private:
    static void customMsgHandler(QtMsgType type, const QMessageLogContext &, const QString &msg);

    std::ostream &m_stream;
    std::streambuf *m_old_buf;
};

class ZWidgetErrStream : public std::basic_streambuf<char>
{
    typedef std::basic_streambuf<char> _base;
public:
    ZWidgetErrStream();
    virtual ~ZWidgetErrStream();
    static void registerMsgHandler();

protected:
    //virtual int_type overflow(int_type);
    virtual std::streamsize xsputn(const char* p, std::streamsize n);
    //This is called when a std::endl has been inserted into the stream
    virtual int_type overflow(int_type v)
    {
        return v;
    }

private:
    static void customMsgHandler(QtMsgType type, const QMessageLogContext &, const QString &msg);
    std::ostream &m_stream;
    std::streambuf *m_old_buf;
};



#endif
