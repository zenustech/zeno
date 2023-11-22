#include "zsnetthread.h"
#include "zsinstance.h"
#define CURL_STATICLIB
#include <curl/curl.h>

#include "zsinstance.h"

size_t writeToString(void* ptr, size_t size, size_t count, void* stream)
{
    ((std::string*)stream)->append((char*)ptr, 0, size * count);
    return size * count;
}

ZsNetThread::ZsNetThread(QObject *parent)
    : QThread{parent}
{

}

void ZsNetThread::setParam(const QString &id, int type, QString url, QByteArray reqData)
{
    m_id = id;
    m_reqType = type;
    m_url = url;
    m_reqData = reqData;
}

void ZsNetThread::run()
{
    switch (m_reqType)
    {
    case 0:netGet(); break;
    default:
        break;
    }
}

void ZsNetThread::netGet()
{
#ifdef __linux__
    return;
#else
    CURL* curl;
    CURLcode res;
    std::string strData;
    QString data;
    curl = curl_easy_init();
    if (curl)
    {
        curl_slist* pHeadlist = NULL;
        pHeadlist = curl_slist_append(pHeadlist, "Content-Type:application/json;charset=UTF-8");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, pHeadlist);
        curl_easy_setopt(curl, CURLOPT_URL, m_url.toStdString().c_str());
        //curl_easy_setopt(curl, CURLOPT_POSTFIELDS, strBody.c_str());//POST参数

        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeToString);//对返回的数据进行操作的函数地址
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &strData);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, false);

        curl_easy_setopt(curl, CURLOPT_POST, 0);   // 0 是get ，1 是post
        res = curl_easy_perform(curl);

        /* Check for errors */

        if (res != CURLE_OK)
            data = QString("{\"code\":-1,\"message\":\"network error:%1\"}").arg(curl_easy_strerror(res));
        else
            data = strData.c_str();

        curl_easy_cleanup(curl);
        curl_slist_free_all(pHeadlist);
    }

    emit netReqFinish(data, m_id);
#endif
}