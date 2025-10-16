#include "pythonexecutor.h"
#include "layout/docktabcontent.h"
#include <zeno/include/zenoutil.h>
#include <zenoapplication.h>
#include "zassert.h"
#include "model/graphsmanager.h"


PythonExecutePane::PythonExecutePane(QWidget* parent)
    : QWidget(parent)
{
    QVBoxLayout* mainLayout = new QVBoxLayout(this);

    // 创建标题栏
    QHBoxLayout* titleBarLayout = new QHBoxLayout();

    auto pBtnShowList = new ZToolBarButton(true, ":/icons/subnet-listview.svg", ":/icons/subnet-listview-on.svg");
    QPushButton* runBtn = new QPushButton("Run");

    pBtnShowList->setChecked(true);

    titleBarLayout->addWidget(pBtnShowList);
    titleBarLayout->addStretch();
    titleBarLayout->addWidget(runBtn);

    mainLayout->addLayout(titleBarLayout);

    // 创建主内容区域
    QHBoxLayout* contentLayout = new QHBoxLayout();
    listWidget = new QListWidget();
    listWidget->setStyleSheet("QListWidget { border: 1px solid gray; }");
    listWidget->setFixedWidth(150);
    stackedWidget = new QStackedWidget();

    contentLayout->addWidget(listWidget);
    contentLayout->addWidget(stackedWidget, 1);

    mainLayout->addLayout(contentLayout);

    QString name = QString("Session %1").arg(1);
    listWidget->addItem(name);
    QTextEdit* textEdit = new QTextEdit();
    stackedWidget->addWidget(textEdit);

    setAutoFillBackground(true);
    QPalette pal = palette();
    pal.setColor(QPalette::Window, QColor(31,31,31));
    setPalette(pal);

    connect(runBtn, &QPushButton::clicked, this, &PythonExecutePane::run);
    connect(listWidget, &QListWidget::currentRowChanged, stackedWidget, &QStackedWidget::setCurrentIndex);
    connect(pBtnShowList, &ZToolBarButton::toggled, listWidget, &QListWidget::setVisible);
}

void PythonExecutePane::run() {
    GraphsManager* graphsMgr = zenoApp->graphsManager();
    if (!graphsMgr->getGraph({ "main" })) {
        QMessageBox::critical(this, "Run Failed", "the project has not been created.");
        return;
    }

    QTextEdit* codeEditor = qobject_cast<QTextEdit*>(stackedWidget->currentWidget());
    bool bSucceed = false;
    if (codeEditor) {
        QString text = codeEditor->toPlainText();
        if (text.isEmpty())
            return;

        const std::string& script = text.toStdString();
#ifdef ZENO_WITH_PYTHON
        bSucceed = zeno::runPython(script);
#endif
    }
    if (!bSucceed) {
        QMessageBox::critical(this, "Run Failed", "The python script run failed, please read the log from log panel.");
    }
    else {
        int nExists = listWidget->count();
        QString name = QString("Session %1").arg(nExists + 1);
        listWidget->addItem(name);
        QTextEdit* textEdit = new QTextEdit();
        stackedWidget->addWidget(textEdit);
        stackedWidget->setCurrentWidget(textEdit);
        listWidget->setCurrentRow(listWidget->count() - 1);
    }
}


PythonAIDialog::PythonAIDialog(QWidget* parent)
    : QDialog(parent)
    , m_clientSocket(new QTcpSocket(this))
{
    setWindowTitle("AI agent");

    QVBoxLayout* layout = new QVBoxLayout(this);

    QLabel* label = new QLabel("Tell me what you want:", this);
    QTextEdit* lineEdit = new QTextEdit(this);
    lineEdit->setText(QStringLiteral("使用我提供的工具api，先给出创建一个长10，宽20，高30的cube的代码。然后对这个节点打上view，最后移除这个节点。"));
    QPushButton* sendButton = new QPushButton("send", this);
    QPushButton* cancelButton = new QPushButton("cancel", this);

    QHBoxLayout* buttonLayout = new QHBoxLayout();
    buttonLayout->addStretch();
    buttonLayout->addWidget(sendButton);
    buttonLayout->addWidget(cancelButton);

    layout->addWidget(label);
    layout->addWidget(lineEdit);
    layout->addLayout(buttonLayout);

    connect(sendButton, &QPushButton::clicked, this, [=]() {
        QString prompt = lineEdit->toPlainText();
        m_clientSocket->connectToHost(QHostAddress::LocalHost, 12306);
        if (!m_clientSocket->waitForConnected(4000)) {
            reject();
            return;
        }

        std::string sPrompt = prompt.toStdString();
        m_clientSocket->write(sPrompt.c_str(), sPrompt.size());
        while (m_clientSocket->bytesToWrite() > 0) {
            m_clientSocket->waitForBytesWritten();
        }

        connect(m_clientSocket, &QTcpSocket::readyRead, [&]() {
            QByteArray arr = m_clientSocket->readAll();
            qint64 redSize = arr.size();
            QString receive = QString::fromUtf8(arr, redSize);
            QMessageBox::information(this, "Connection", receive);
            accept();
            });
    });

    connect(cancelButton, &QPushButton::clicked, this, &QDialog::reject);
}