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
    palette.setColor(QPalette::Background, QColor("#2d3239"));
    setPalette(palette);

    QSize iconSize(ZenoStyle::dpiScaled(24), ZenoStyle::dpiScaled(24));
    m_ui->btnNew->setIconSize(iconSize);
    m_ui->btnOpen->setIconSize(iconSize);
    m_ui->btnNew->setIcon(QIcon(":/icons/file_newfile.svg"));
    m_ui->btnOpen->setIcon(QIcon(":/icons/file_openfile.svg"));
	m_ui->btnNew->setProperty("cssClass", "welcomepage");
    m_ui->btnOpen->setProperty("cssClass", "welcomepage");
    m_ui->lblStart->setProperty("cssClass", "welcomepage");
    m_ui->lblRecentFiles->setProperty("cssClass", "welcomepage");

    QSize size(ZenoStyle::dpiScaled(20), ZenoStyle::dpiScaled(20));

    QMargins margin(ZenoStyle::dpiScaled(10), ZenoStyle::dpiScaled(7), ZenoStyle::dpiScaled(10), ZenoStyle::dpiScaled(7));

	initSignals();
}

void ZenoWelcomePage::initSignals()
{
	connect(m_ui->btnNew, SIGNAL(clicked()), this, SIGNAL(newRequest()));
	connect(m_ui->btnOpen, SIGNAL(clicked()), this, SIGNAL(openRequest()));

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
    QLayoutItem *child;
    while ((child = layout->takeAt(0)) != nullptr) {
        if (child->widget()) {
            child->widget()->setParent(nullptr);
            child->widget()->deleteLater();
        } else if (child->layout()) {
            deleteItem(child->layout());
            child->layout()->deleteLater();
        }
        delete child;
    }
}
