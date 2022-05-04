#ifndef __ZWIDGET_OSTREAM_H__
#define __ZWIDGET_OSTREAM_H__

#include <iostream>
#include <QtWidgets>

class ZWidgetErrStream : public std::basic_streambuf<char>
{
    typedef std::basic_streambuf<char> _base;
public:
    ZWidgetErrStream();
    virtual ~ZWidgetErrStream();
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

#endif
