#include "zpathedit.h"
#include "zlineedit.h"
#include "uicommon.h"
#include <zeno/extra/TempNode.h>
#include <zeno/extra/assetDir.h>
#include "zenomainwindow.h"
#include "zenoapplication.h"
#include "model/graphsmanager.h"

ZPathEdit::ZPathEdit(zeno::ParamControl ctrl, QWidget *parent)
    : ZLineEdit(parent)
    , m_ctrl(ctrl)
{
    initUI();
}

ZPathEdit::ZPathEdit(const QString &text, zeno::ParamControl ctrl, QWidget *parent)
    : ZLineEdit(text, parent)
    , m_ctrl(ctrl)
{
    initUI();
}

void ZPathEdit::setPathFlag(zeno::ParamControl ctrl)
{
    m_ctrl = ctrl;
}

void ZPathEdit::initUI()
{
    setFocusPolicy(Qt::ClickFocus);
    setIcons(":/icons/file-loader.svg", ":/icons/file-loader-on.svg");
    setProperty("cssClass", "path_edit");

    QObject::connect(this, &ZLineEdit::btnClicked, [=]() {
        QString path = this->text();

        QString zsgDir = zenoApp->graphsManager()->zsgDir();
        QString filePath = path;

        // need to resolve the formula path
        {
            zeno::setConfigVariable("ZSG", zsgDir.toStdString());
            auto code = std::make_shared<zeno::StringObject>();
            code->set(path.toStdString());
            auto outs = zeno::TempNodeSimpleCaller("StringEval")
                .set("zfxCode", code)
                .call();
            std::shared_ptr<zeno::StringObject> spStrObj = outs.get<zeno::StringObject>("result");
            if (spStrObj)
            {
                filePath = QString::fromStdString(spStrObj->get());
            }
        }

        QString dirPath;

        if (filePath.isEmpty()) {
            dirPath = zsgDir;
        }
        else {
            QFileInfo fileInfo(filePath);
            QDir dir = fileInfo.dir();
            dirPath = dir.path();
        }
        if (m_ctrl == zeno::ReadPathEdit) {
            path = QFileDialog::getOpenFileName(nullptr, "File to Open", dirPath, "All Files(*);;");
        } 
        else if (m_ctrl == zeno::WritePathEdit) {
            path = QFileDialog::getSaveFileName(nullptr, "Path to Save", dirPath, "All Files(*);;");
        }
        else {
            path = QFileDialog::getExistingDirectory(nullptr, "Path to Save", "");
        }
        if (path.isEmpty()) {
            zenoApp->getMainWindow()->setInDlgEventLoop(false);
            return;
        }
        if (!zsgDir.isEmpty() && path.indexOf(zsgDir) != -1)
            path.replace(zsgDir, "=$ZSG");
        setText(path);
        emit textEditFinished();
        clearFocus();
        zenoApp->getMainWindow()->setInDlgEventLoop(false);
    });
}
