#include "ui_welcomepage.h"
#include "zenowelcomepage.h"
#include <QtWidgets>
#include <zenoui/comctrl/zlabel.h>
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "settings/zsettings.h"


ZenoWelcomePage::ZenoWelcomePage(QWidget* parent)
	: QWidget(parent)
{
	m_ui = new Ui::WelcomePage;
	m_ui->setupUi(this);

	m_ui->btnNew->setProperty("cssClass", "welcomepage");
    m_ui->btnOpen->setProperty("cssClass", "welcomepage");
    m_ui->lblCurrVer->setProperty("cssClass", "welcomepage");
    m_ui->lblLink->setProperty("cssClass", "welcomepage");
    m_ui->lblWhatsNew->setProperty("cssClass", "welcomepage");
    m_ui->lblRecentFiles->setProperty("cssClass", "welcomepage");
    m_ui->label->setProperty("cssClass", "welcomepage_label");
    m_ui->lblLogo->setProperty("cssClass", "welcomepage_name");

    m_ui->lblManual->setText(tr("ZENO Manual"));
    m_ui->lblManual->setFont(QFont("HarmonyOS Sans", 11));
    m_ui->lblManual->setTextColor(QColor(133, 130, 128));
    m_ui->lblManual->setUnderlineOnHover(true);

    m_ui->lblVideos->setText(tr("ZENO Video Tutorials"));
    m_ui->lblVideos->setFont(QFont("HarmonyOS Sans", 11));
    m_ui->lblVideos->setTextColor(QColor(133, 130, 128));
    m_ui->lblVideos->setUnderlineOnHover(true);

    m_ui->lblOfficialWeb->setText(tr("Zenus Official Web"));
    m_ui->lblOfficialWeb->setFont(QFont("HarmonyOS Sans", 11));
    m_ui->lblOfficialWeb->setTextColor(QColor(133, 130, 128));
    m_ui->lblOfficialWeb->setUnderlineOnHover(true);

    m_ui->lblForum->setText(tr("Forum"));
    m_ui->lblForum->setFont(QFont("HarmonyOS Sans", 11));
    m_ui->lblForum->setTextColor(QColor(133, 130, 128));
    m_ui->lblForum->setUnderlineOnHover(true);

    m_ui->lblGithub->setText(tr("Project on GitHub"));
    m_ui->lblGithub->setFont(QFont("HarmonyOS Sans", 11));
    m_ui->lblGithub->setTextColor(QColor(133, 130, 128));
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
                QWidget *widget = item->widget();
                pLabel = qobject_cast<ZTextLabel *>(widget);
                if (pLabel)
                    pLabel->setText(fn);
            } 
            else {
                pLabel = new ZTextLabel(fn);
                pLabel->setTextColor(QColor(133, 130, 128));
                pLabel->setFont(QFont("HarmonyOS Sans", 11));
                pLabel->setToolTip(path);

                m_ui->layoutFiles->addWidget(pLabel);
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
