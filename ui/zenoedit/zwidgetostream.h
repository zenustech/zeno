#ifndef __ZWIDGET_OSTREAM_H__
#define __ZWIDGET_OSTREAM_H__

#include <iostream>
#include <QtWidgets>
#include <QString>
#include <string>

class ZWidgetErrStream : public std::basic_streambuf<char>
{
    typedef std::basic_streambuf<char> _base;
public:
    explicit ZWidgetErrStream(std::ostream &stream);
    virtual ~ZWidgetErrStream();
    static void registerMsgHandler();
    static void appendFormatMsg(std::string const& str);

protected:
    virtual std::streamsize xsputn(const char* p, std::streamsize n) override;
    virtual int_type overflow(int_type v) override
    {
        return v;
    }

private:
    static void customMsgHandler(QtMsgType type, const QMessageLogContext &, const QString &msg);
    static bool isGUIThread();

    std::ostream &m_stream;
    std::streambuf *m_old_buf;
    std::string m_linebuffer;
};

#endif
