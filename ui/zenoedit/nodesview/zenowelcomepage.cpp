#include "ui_welcomepage.h"
#include "zenowelcomepage.h"
#include <QtWidgets>
#include <zenoui/comctrl/zlabel.h>
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "settings/zsettings.h"
#include <zenoui/style/zenostyle.h>
#include <QtSvg/QSvgWidget>
#include "startup/zstartup.h"


ZenoWelcomePage::ZenoWelcomePage(QWidget* parent)
	: QWidget(parent)
{
	m_ui = new Ui::WelcomePage;
	m_ui->setupUi(this);
    QPalette palette = this->palette();
    palette.setColor(QPalette::Background, QColor(44,50,58));
    setPalette(palette);

    QSize iconSize(ZenoStyle::dpiScaled(24), ZenoStyle::dpiScaled(24));
    m_ui->btnNew->setIconSize(iconSize);
    m_ui->btnOpen->setIconSize(iconSize);
    m_ui->btnNew->setIcon(QIcon(":/icons/file_newfile.svg"));
    m_ui->btnOpen->setIcon(QIcon(":/icons/file_openfile.svg"));
	m_ui->btnNew->setProperty("cssClass", "welcomepage");
    m_ui->btnOpen->setProperty("cssClass", "welcomepage");
    m_ui->lblCurrVer->setProperty("cssClass", "welcomepage");
    m_ui->lblLink->setProperty("cssClass", "welcomepage");
    m_ui->lblStart->setProperty("cssClass", "welcomepage");
    m_ui->lblRecentFiles->setProperty("cssClass", "welcomepage");
    m_ui->lblLogo->setProperty("cssClass", "welcomepage_name");

    m_ui->widgetManual->setProperty("cssClass", "welcomepage_link");
    m_ui->widgetVideos->setProperty("cssClass", "welcomepage_link");
    m_ui->widgetFromNum->setProperty("cssClass", "welcomepage_link");
    m_ui->widgetOfficialWeb->setProperty("cssClass", "welcomepage_link");
    m_ui->widgetGitHub->setProperty("cssClass", "welcomepage_link");

    m_ui->lblLogoIcon->load(QString(":/icons/welcome_Zeno_logo.svg"));
    m_ui->lblLogoIcon->setFixedSize(QSize(ZenoStyle::dpiScaled(60), ZenoStyle::dpiScaled(60)));

    QSize size(ZenoStyle::dpiScaled(20), ZenoStyle::dpiScaled(20));
    m_ui->iconManual->setFixedSize(size);
    m_ui->iconVideos->setFixedSize(size);
    m_ui->iconForum->setFixedSize(size);
    m_ui->iconOfficialWeb->setFixedSize(size);
    m_ui->iconGithub->setFixedSize(size);
    QMargins margin(ZenoStyle::dpiScaled(10), ZenoStyle::dpiScaled(7), ZenoStyle::dpiScaled(10), ZenoStyle::dpiScaled(7));
    m_ui->widgetManual->setContentsMargins(margin);
    m_ui->widgetVideos->setContentsMargins(margin);
    m_ui->widgetFromNum->setContentsMargins(margin);
    m_ui->widgetOfficialWeb->setContentsMargins(margin);
    m_ui->widgetGitHub->setContentsMargins(margin);
    m_ui->widgetVideos->hide();

    m_ui->lblManual->setText(tr("ZENO Manual"));
    m_ui->lblManual->setProperty("cssClass", "welcomepage_label");
    m_ui->lblManual->setUnderlineOnHover(true);

    m_ui->lblVideos->setText(tr("ZENO Video Tutorials"));
    m_ui->lblVideos->setProperty("cssClass", "welcomepage_label");
    m_ui->lblVideos->setUnderlineOnHover(true);

    m_ui->lblOfficialWeb->setText(tr("Zenus Official Web"));
    m_ui->lblOfficialWeb->setProperty("cssClass", "welcomepage_label");
    m_ui->lblOfficialWeb->setUnderlineOnHover(true);
 
    m_ui->lblCurrVer->setText(QString::fromStdString(getZenoVersion()));

    m_ui->lblForum->setText(tr("Forum"));
    m_ui->lblForum->setProperty("cssClass", "welcomepage_label");
    m_ui->lblForum->setUnderlineOnHover(true);

    m_ui->lblGithub->setText(tr("Project on GitHub"));
    m_ui->lblGithub->setProperty("cssClass", "welcomepage_label");
    m_ui->lblGithub->setUnderlineOnHover(true);

	initSignals();
}

void ZenoWelcomePage::initSignals()
{
	connect(m_ui->btnNew, SIGNAL(clicked()), this, SIGNAL(newRequest()));
	connect(m_ui->btnOpen, SIGNAL(clicked()), this, SIGNAL(openRequest()));

    connect(m_ui->lblManual, &ZTextLabel::clicked, this, [=]() {
        QDesktopServices::openUrl(QUrl("https://doc.zenustech.com/"));
    });
    connect(m_ui->lblVideos, &ZTextLabel::clicked, this, [=]() {
        QDesktopServices::openUrl(QUrl("https://zenustech.com/learn"));
    });
    connect(m_ui->lblOfficialWeb, &ZTextLabel::clicked, this, [=]() {
        QDesktopServices::openUrl(QUrl("https://zenustech.com/"));
    });
    connect(m_ui->lblForum, &ZTextLabel::clicked, this, [=]() {
        QDesktopServices::openUrl(QUrl("https://forums.zenustech.com/"));
    });
    connect(m_ui->lblGithub, &ZTextLabel::clicked, this, [=]() {
        QDesktopServices::openUrl(QUrl("https://github.com/zenustech/zeno"));
    });

    connect(zenoApp->getMainWindow(), &ZenoMainWindow::recentFilesChanged, this, [=](const QObject *sender) {
        if (sender != this) {
            initRecentFiles();
        }
    });
}

void ZenoWelcomePage::initRecentFiles()
{
    QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
    settings.beginGroup("Recent File List");
    QStringList lst = settings.childKeys();
    zenoApp->getMainWindow()->sortRecentFile(lst);
    while (m_ui->layoutFiles->count() > 0) {
        QLayoutItem *item = m_ui->layoutFiles->itemAt(0);
        deleteItem(item->layout());
        m_ui->layoutFiles->removeItem(item);
        delete item;
        item = nullptr;
    }
    for (int i = 0; i < lst.size(); i++)
    {
        const QString& key = lst[i];
        const QString& path = settings.value(key).toString();
        if (!path.isEmpty())
        {
            QFileInfo info(path);
            const QString& fn = info.fileName();
            QLayoutItem *item;
            ZTextLabel *pLabel;
            QLayout *layout = nullptr;
            if (item = m_ui->layoutFiles->itemAt(i)) {
                layout = item->layout();
                if (!layout)
                    return;
                QWidget *widget = layout->itemAt(1)->widget();
                pLabel = qobject_cast<ZTextLabel *>(widget);
                if (pLabel)
                    pLabel->setText(fn);
            } 
            else {
                layout = new QHBoxLayout;
                QSvgWidget* iconLabel = new QSvgWidget(":/icons/file_zsgfile.svg");
                iconLabel->setFixedSize(QSize(ZenoStyle::dpiScaled(24), ZenoStyle::dpiScaled(24)));
                layout->addWidget(iconLabel);
                pLabel = new ZTextLabel(fn);
                pLabel->setTextColor(QColor("#A3B1C0"));
                pLabel->setProperty("cssClass", "welcomepage_label");
                pLabel->setToolTip(path);

                layout->addWidget(pLabel);
                m_ui->layoutFiles->addLayout(layout);
            }

            connect(pLabel, &ZTextLabel::clicked, this, [=]() {

                if (!QFileInfo::exists(path)) {
                    int flag = QMessageBox::question(nullptr, "", tr("the file does not exies, do you want to remove it?"), QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
                    if (flag & QMessageBox::Yes)
                    {
                        QSettings _settings(QSettings::UserScope, zsCompanyName, zsEditor);
                        _settings.beginGroup("Recent File List");
                        _settings.remove(key);
                        deleteItem(layout);
                        m_ui->layoutFiles->removeItem(layout);
                        emit zenoApp->getMainWindow() ->recentFilesChanged(this);
                    }
                } else {
                    zenoApp->getMainWindow()->openFile(path);
                }
            });

            connect(pLabel, &ZTextLabel::rightClicked, this, [=]() {
                QMenu* pMenu = new QMenu(this);
                QAction *pDelete = new QAction(tr("Remove"));
                QAction *pOpen= new QAction(tr("Open file location"));
                pMenu->addAction(pDelete);
                pMenu->addAction(pOpen);
                connect(pDelete, &QAction::triggered, this, [=]() {
                    QSettings _settings(QSettings::UserScope, zsCompanyName, zsEditor);
                    _settings.beginGroup("Recent File List");
                    _settings.remove(key);
                    deleteItem(layout);
                    m_ui->layoutFiles->removeItem(layout);
                    emit zenoApp->getMainWindow()->recentFilesChanged(this);
                });

                connect(pOpen, &QAction::triggered, this, [=]() {
                    if (!QFileInfo::exists(path)) 
                    {
                        QMessageBox::information(this, "", tr("The file does not exist!"));
                        return;
                    }
                    QString filePath = path;
                    QString cmd;
                    #ifdef _WIN32
                    filePath = filePath.replace("/", "\\");
                    cmd = QString("explorer.exe /select,%1").arg(filePath);
                    #else
                    filePath = filePath.replace("\\", "/");
                    cmd = QString("open -R %1").arg(filePath);
                    #endif
                    QProcess process;
                    process.startDetached(cmd);
                });
                pMenu->exec(QCursor::pos());
                pMenu->deleteLater();
            });
        }
    }
}

void ZenoWelcomePage::deleteItem(QLayout *layout) {
    if (!layout)
        return;
    while (QLayoutItem * child = layout->takeAt(0)) {
        if (child->widget()) {
            child->widget()->setParent(nullptr);
            child->widget()->deleteLater();
        } else if (child->layout()) {
            deleteItem(child->layout());
            child->layout()->deleteLater();
        }
        delete child;
        child = nullptr;
    }
}
