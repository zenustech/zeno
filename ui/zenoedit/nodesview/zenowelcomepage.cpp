#include "ui_welcomepage.h"
#include "zenowelcomepage.h"
#include <QtWidgets>
#include <zenoui/comctrl/zlabel.h>
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "settings/zsettings.h"
#include <zenoui/style/zenostyle.h>


ZenoWelcomePage::ZenoWelcomePage(QWidget* parent)
	: QWidget(parent)
{
	m_ui = new Ui::WelcomePage;
	m_ui->setupUi(this);
    QPalette palette = this->palette();
    palette.setColor(QPalette::Background, QColor(44,50,58));
    setPalette(palette);

	m_ui->btnNew->setProperty("cssClass", "welcomepage");
    m_ui->btnOpen->setProperty("cssClass", "welcomepage");
    m_ui->lblCurrVer->setProperty("cssClass", "welcomepage");
    m_ui->lblLink->setProperty("cssClass", "welcomepage");
    m_ui->lblStart->setProperty("cssClass", "welcomepage");
    m_ui->lblRecentFiles->setProperty("cssClass", "welcomepage");
    m_ui->lblLogo->setProperty("cssClass", "welcomepage_name");
    m_ui->lblIogoIcon->setProperty("cssClass", "welcomepage_logo");
    m_ui->widgetManual->setProperty("cssClass", "welcomepage_link");
    m_ui->widgetVideos->setProperty("cssClass", "welcomepage_link");
    m_ui->widgetFromNum->setProperty("cssClass", "welcomepage_link");
    m_ui->widgetOfficialWeb->setProperty("cssClass", "welcomepage_link");
    m_ui->widgetGitHub->setProperty("cssClass", "welcomepage_link");
    m_ui->lblIogoIcon->setFixedSize(QSize(ZenoStyle::dpiScaled(24), ZenoStyle::dpiScaled(24)));
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
 

    m_ui->lblForum->setText(tr("Forum"));
    m_ui->lblForum->setProperty("cssClass", "welcomepage_label");
    m_ui->lblForum->setUnderlineOnHover(true);

    m_ui->lblGithub->setText(tr("Project on GitHub"));
    m_ui->lblGithub->setProperty("cssClass", "welcomepage_label");
    m_ui->lblGithub->setUnderlineOnHover(true);

    m_ui->widgetCenter->setFixedWidth(ZenoStyle::dpiScaled(315));
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

    connect(zenoApp->getMainWindow(), &ZenoMainWindow::recentFilesChanged, this, [=]() {
        initRecentFiles();
    });
}

void ZenoWelcomePage::initRecentFiles()
{
    QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
    settings.beginGroup("Recent File List");
    QStringList lst = settings.childKeys();
    zenoApp->getMainWindow()->sortRecentFile(lst);

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
            if (item = m_ui->layoutFiles->itemAt(i)) {
                QLayout *layout = item->layout();
                if (!layout)
                    return;
                QWidget *widget = layout->itemAt(1)->widget();
                pLabel = qobject_cast<ZTextLabel *>(widget);
                if (pLabel)
                    pLabel->setText(fn);
            } 
            else {
                QHBoxLayout *layout = new QHBoxLayout(this);
                QLabel *iconLabel = new QLabel(this);
                iconLabel->setFixedSize(QSize(ZenoStyle::dpiScaled(16), ZenoStyle::dpiScaled(16)));
                iconLabel->setPixmap(QPixmap(":/icons/file_zsgfile.svg"));
                layout->addWidget(iconLabel);
                pLabel = new ZTextLabel(fn);
                pLabel->setTextColor(QColor("#A3B1C0"));
                pLabel->setProperty("cssClass", "welcomepage_label");
                pLabel->setToolTip(path);

                 layout->addWidget(pLabel);
                m_ui->layoutFiles->addLayout(layout);
            }

            connect(pLabel, &ZTextLabel::clicked, this, [=]() {
                bool ret = zenoApp->getMainWindow()->openFile(path);
                if (!ret) {
                    int flag = QMessageBox::question(nullptr, "", tr("the file does not exies, do you want to remove it?"), QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
                    if (flag & QMessageBox::Yes)
                    {
                        QSettings _settings(QSettings::UserScope, zsCompanyName, zsEditor);
                        _settings.beginGroup("Recent File List");
                        _settings.remove(key);
                        m_ui->layoutFiles->removeWidget(pLabel);
                        pLabel->deleteLater();
                    }
                }
            });

            connect(pLabel, &ZTextLabel::rightClicked, this, [=]() {
                QMenu* pMenu = new QMenu(this);
                QAction *pDelete = new QAction(tr("Remove"));
                pMenu->addAction(pDelete);
                connect(pDelete, &QAction::triggered, this, [=]() {
                    QSettings _settings(QSettings::UserScope, zsCompanyName, zsEditor);
                    _settings.beginGroup("Recent File List");
                    _settings.remove(key);
                    m_ui->layoutFiles->removeWidget(pLabel);
                    pLabel->deleteLater();
                });
                pMenu->exec(QCursor::pos());
                pMenu->deleteLater();
            });
        }
    }
}
