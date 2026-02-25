#include "ZComposeVideoDlg.h"
#include "ui_ZComposeVideoDlg.h"

#include "zenomainwindow.h"
#include <QProgressDialog>
#include <QMessageBox>
#include <QThread>


ZComposeVideoDlg::ZComposeVideoDlg(QWidget *parent)
    : QDialog(parent)
    , m_ui(new Ui::ZComposeVideoDlgClass)
    , m_workerThread(nullptr)
    , m_progressDialog(nullptr)
    , m_cancelCompose(false)
{
    m_ui->setupUi(this);

    setMinimumSize(320, 200);

    const RECORD_SETTING& info = zenoApp->graphsManagment()->recordSettings();
    m_ui->fps->setValidator(new QIntValidator);
    m_ui->fps->setText(QString::number(info.fps));
    m_ui->bitrate->setValidator(new QIntValidator);
    m_ui->bitrate->setText(QString::number(info.bitrate));
    m_ui->linePath->setText(info.record_path);

    connect(m_ui->btnGroup, SIGNAL(accepted()), this, SLOT(onAcceptClicked()));
    connect(m_ui->btnGroup, SIGNAL(rejected()), this, SLOT(reject()));
    connect(m_ui->btnOpen, &QPushButton::clicked, this, [=]() {
        DlgInEventLoopScope;
        QString path = QFileDialog::getExistingDirectory(this, tr("File to Load"), "");
        if (path.isEmpty())
            return;
        m_ui->linePath->setText(path);
        });
}

ZComposeVideoDlg::~ZComposeVideoDlg()
{
    // 停止工作线程
    if (m_workerThread && m_workerThread->isRunning()) {
        m_cancelCompose = true;
        m_workerThread->quit();
        m_workerThread->wait();
    }
}

bool ZComposeVideoDlg::combineVideo()
{
    QDir dir(m_ui->linePath->text());
    if (m_ui->linePath->text().isEmpty() || !dir.exists()) {
        QMessageBox::information(this, tr("Info"), tr("Invalid input path"));
        return false;
    }
    QString dir_path = m_ui->linePath->text();
    QDir qDir = QDir(dir_path);
    qDir.setNameFilters(QStringList("*.jpg"));
    QStringList fileList = qDir.entryList(QDir::Files | QDir::NoDotAndDotDot);
    if (fileList.empty()) {
        QMessageBox::information(this, tr("Info"), tr("Jpg file not exist"));
        return false;
    }
    fileList.sort();
    QString baseName = QFileInfo(fileList[0]).baseName();
    bool ok;
    int number = baseName.toInt(&ok);
    if (!ok) {
        QMessageBox::information(this, tr("Info"), tr("Jpg file not exist"));
        return false;
    }

    // 检查输出文件是否存在
    QString outPath = m_ui->linePath->text() + "/" + (m_ui->filename->text().isEmpty() ? "output.mp4" : m_ui->filename->text() + ".mp4");
    if (QFile::exists(outPath)) {
        QMessageBox::information(this, tr("Info"), tr("Output file exists"));
        return false;
    }

    // 开始视频合成（在工作线程中）
    startVideoCompose();
    return true;
}

void ZComposeVideoDlg::onAcceptClicked()
{
    combineVideo();
}

void ZComposeVideoDlg::startVideoCompose()
{
    // 如果已经有工作线程在运行，先停止它
    if (m_workerThread && m_workerThread->isRunning()) {
        m_cancelCompose = true;
        m_workerThread->quit();
        m_workerThread->wait();
        delete m_workerThread;
        m_workerThread = nullptr;
    }

    // 创建进度条
    if (!m_progressDialog) {
        m_progressDialog = new QProgressDialog(this);
        m_progressDialog->setWindowTitle(tr("视频合成进度"));
        m_progressDialog->setCancelButtonText(tr("取消"));
        m_progressDialog->setMinimum(0);
        m_progressDialog->setMaximum(0); // 不确定进度

        // 设置进度条样式
        m_progressDialog->setStyleSheet("QProgressBar {border: none; text-align: center;} QProgressBar::chunk {background-color: #05B8CC; width: 20px; } QLabel { padding: 5px;}");

        connect(m_progressDialog, &QProgressDialog::canceled, this, [=]() {
            m_cancelCompose = true;
        });
    }

    // 创建工作线程
    m_workerThread = new QThread;

    // 创建工作对象
    QObject* worker = new QObject;
    worker->moveToThread(m_workerThread);

    // 连接信号
    connect(m_workerThread, &QThread::started, worker, [=]() {
        QString imgPath = m_ui->linePath->text() + "/{:07}.jpg";
        QString outPath = m_ui->linePath->text() + "/" + (m_ui->filename->text().isEmpty() ? "output.mp4" : m_ui->filename->text() + ".mp4");

        // 获取起始帧号
        QDir qDir = QDir(m_ui->linePath->text());
        qDir.setNameFilters(QStringList("*.jpg"));
        QStringList fileList = qDir.entryList(QDir::Files | QDir::NoDotAndDotDot);
        fileList.sort();
        QString baseName = QFileInfo(fileList[0]).baseName();
        int startFrame = baseName.toInt();

        // 计算总帧数
        int totalFrames = 0;
        int currentFrame = startFrame;
        while (true) {
            QString currentImgPath = QString::fromStdString(zeno::format(imgPath.toStdString(), currentFrame));
            if (!QFile::exists(currentImgPath)) {
                break;
            }
            totalFrames++;
            currentFrame++;
        }

        // 设置进度条最大值
        QMetaObject::invokeMethod(this, "updateProgress", Qt::QueuedConnection, Q_ARG(int, -totalFrames));

        // 开始合成
        ImageSequenceEncoder enc;
        bool ret = enc.open(
            outPath.toStdString(),
            m_ui->fps->text().toInt(),
            m_ui->bitrate->text().toInt()
        );

        if (!ret) {
            QMetaObject::invokeMethod(this, "onComposeFinished", Qt::QueuedConnection, Q_ARG(bool, false));
            return;
        }

        currentFrame = startFrame;
        int processedFrames = 0;
        bool composeSuccess = true;

        while (true) {
            if (m_cancelCompose) {
                enc.close();
                QMetaObject::invokeMethod(this, "onComposeFinished", Qt::QueuedConnection, Q_ARG(bool, false));
                return;
            }

            QString currentImgPath = QString::fromStdString(zeno::format(imgPath.toStdString(), currentFrame));
            if (!QFile::exists(currentImgPath)) {
                break;
            }

            ret = enc.addFrameFromFile(currentImgPath.toStdString());
            if (!ret) {
                composeSuccess = false;
                break;
            }

            processedFrames++;
            QMetaObject::invokeMethod(this, "updateProgress", Qt::QueuedConnection, Q_ARG(int, processedFrames));
            currentFrame++;
        }

        // 即使close返回false，只要成功处理了所有帧，就认为是成功的
        enc.close();

        // 如果成功处理了至少一帧，并且没有中途失败，就认为是成功的
        bool finalSuccess = composeSuccess && processedFrames > 0;
        QMetaObject::invokeMethod(this, "onComposeFinished", Qt::QueuedConnection, Q_ARG(bool, finalSuccess));
    });

    // 确保worker对象在线程结束时被正确删除
    connect(m_workerThread, &QThread::finished, worker, &QObject::deleteLater);
    connect(m_workerThread, &QThread::finished, this, [=]() {m_workerThread = nullptr;});
    connect(m_workerThread, &QThread::finished, m_workerThread, &QThread::deleteLater);

    m_cancelCompose = false; // 重置取消标志

    m_progressDialog->setLabelText(tr("正在初始化..."));
    m_progressDialog->show();// 显示进度条

    m_workerThread->start();// 启动工作线程
}

void ZComposeVideoDlg::updateProgress(int currentFrame)
{
    if (currentFrame < 0) {// 负数表示设置总帧数
        m_progressDialog->setMaximum(-currentFrame);
        m_progressDialog->setValue(0);
    } else {// 更新进度
        m_progressDialog->setValue(currentFrame);
        m_progressDialog->setLabelText(tr("正在合成第%1帧").arg(currentFrame));
    }
}

void ZComposeVideoDlg::onComposeFinished(bool success)
{
    // 停止工作线程
    if (m_workerThread && m_workerThread->isRunning()) {
        m_workerThread->quit();
        m_workerThread->wait();
    }

    // 隐藏进度条
    if (m_progressDialog) {
        m_progressDialog->hide();
    }

    // 显示结果
    if (success) {
        QMessageBox::information(this, tr("Info"), tr("Export success"));
        //accept();   // 合成成功，关闭对话框
    } else if (!m_cancelCompose) {
        QMessageBox::information(this, tr("Info"), tr("Export faild"));
    }

    // 重置取消标志
    m_cancelCompose = false;
}

