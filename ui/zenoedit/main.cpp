#include <QApplication>
#include <zeno/extra/assetDir.h>
#include "style/zenostyle.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "startup/zstartup.h"
#include "panel/zdicteditor.h"
#include <zenoui/model/dictmodel.h>


int main(int argc, char *argv[]) 
{
    startUp();
#ifdef ZENO_MULTIPROCESS
    if (argc == 3 && !strcmp(argv[1], "-runner")) {
        extern int runner_main(int sessionid);
        return runner_main(atoi(argv[2]));
    }
#endif
    ZenoApplication a(argc, argv);
    a.setStyle(new ZenoStyle);

    zeno::setExecutableDir(a.applicationDirPath().toStdString());

    QTranslator t;
    if (t.load(":languages/zh.qm")) {
        a.installTranslator(&t);
    }

	ZenoMainWindow mainWindow;
    mainWindow.showMaximized();

    //DictModel dictModel;

    //QStandardItem *pItem = new QStandardItem("");
    //pItem->setData(QVariant("a"), ROLE_KEY);
    //pItem->setData(QVariant(3), ROLE_DATATYPE);
    //pItem->setData(QVariant("int"), ROLE_VALUE);
    //dictModel.appendRow(pItem);

    //ZDictEditor dictEditor(&dictModel);
    //dictEditor.show();

	return a.exec();
}
