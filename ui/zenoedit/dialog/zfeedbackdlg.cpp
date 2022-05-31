#include "zfeedbackdlg.h"
#include "ui_zfeedbackdlg.h"
#include <QDesktopServices>
#include <QUrl>
#include <curl/curl.h>


#define FROM_MAIL     "<zenobugreporter@163.com>"
#define TO_MAIL       "<626332185@qq.com>"  //email of luzh
#define CC_MAIL       ""    //todo: email of zenus tech inc.
 
static const char *payload_text =
  "Date: Mon, 29 Nov 2010 21:54:29 +1100\r\n"
  "To: " TO_MAIL "\r\n"
  "From: " FROM_MAIL "\r\n"
  "Message-ID: <dcd7cb36-11db-487a-9f3a-e652a9458efd@"
  "rfcpedant.example.org>\r\n"
  "Subject: SMTP example message\r\n"
  "\r\n" /* empty line to divide headers from body, see RFC5322 */
  "The body of the message starts here.\r\n"
  "\r\n"
  "It could be 233 a lot of lines, could be MIME encoded, whatever.\r\n"
  "Check RFC5322.\r\n";

struct upload_status {
    size_t bytes_read;
};

static size_t payload_source(char *ptr, size_t size, size_t nmemb, void *userp) {
    struct upload_status *upload_ctx = (struct upload_status *)userp;
    const char *data;
    size_t room = size * nmemb;

    if ((size == 0) || (nmemb == 0) || ((size * nmemb) < 1)) {
        return 0;
    }

    data = &payload_text[upload_ctx->bytes_read];

    if (data) {
        size_t len = strlen(data);
        if (room < len)
            len = room;
        memcpy(ptr, data, len);
        upload_ctx->bytes_read += len;

        return len;
    }

    return 0;
}


ZFeedBackDlg::ZFeedBackDlg(QWidget * parent)
    : QDialog(parent)
{
    m_ui = new Ui::FeedBackDlg;
    m_ui->setupUi(this);
}

QString ZFeedBackDlg::content() const
{
    return m_ui->textEdit->toPlainText();
}

bool ZFeedBackDlg::isSendFile() const
{
    return m_ui->cbSendThisFile->isChecked();
}

void ZFeedBackDlg::sendEmail(const QString& subject, const QString& content, const QString& zsgContent)
{
    CURL *curl;
    CURLcode res = CURLE_OK;
    struct curl_slist *recipients = NULL;
    struct upload_status upload_ctx = {0};

    curl = curl_easy_init();
    if (curl) {
        /* Set username and password */
        curl_easy_setopt(curl, CURLOPT_USERNAME, "zenobugreporter");
        //auth pw£ºDKVGFCXPPQLANYWR from 163.com
        curl_easy_setopt(curl, CURLOPT_PASSWORD, "DKVGFCXPPQLANYWR");

        curl_easy_setopt(curl, CURLOPT_URL, "smtp://smtp.163.com");
#ifdef SKIP_PEER_VERIFICATION
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
#endif
#ifdef SKIP_HOSTNAME_VERIFICATION
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
#endif
        curl_easy_setopt(curl, CURLOPT_MAIL_FROM, FROM_MAIL);
        recipients = curl_slist_append(recipients, TO_MAIL);
        recipients = curl_slist_append(recipients, CC_MAIL);
        curl_easy_setopt(curl, CURLOPT_MAIL_RCPT, recipients);

        curl_easy_setopt(curl, CURLOPT_READFUNCTION, payload_source);
        curl_easy_setopt(curl, CURLOPT_READDATA, &upload_ctx);
        curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);

        /* Since the traffic will be encrypted, it is very useful to turn on debug
     * information within libcurl to see what is happening during the
     * transfer */
        curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

        /* Send the message */
        res = curl_easy_perform(curl);
        if (res != CURLE_OK)
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));

        /* Free the list of recipients */
        curl_slist_free_all(recipients);

        /* Always cleanup */
        curl_easy_cleanup(curl);
    }

    //QString msg = QString("mailto:626332185@qq.com?subject=%1&body=%2").arg(subject).arg(content);
    //QDesktopServices::openUrl(QUrl(msg));
}