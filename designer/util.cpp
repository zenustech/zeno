#include "util.h"

QString GenerateRandomString()
{
    const QString possibleCharacters("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789");
    const int randomStringLength = 12;// assuming you want random strings of 12 characters
    qsrand(QDateTime::currentMSecsSinceEpoch() % UINT_MAX);

    QString randomString;
    for (int i = 0; i < randomStringLength; ++i) {
        int index = qrand() % possibleCharacters.length();
        QChar nextChar = possibleCharacters.at(index);
        randomString.append(nextChar);
    }
    return randomString;
}

DesignerMainWin* getMainWindow()
{
    foreach (QWidget *w, qApp->topLevelWidgets())
        if (DesignerMainWin *mainWin = qobject_cast<DesignerMainWin *>(w))
            return mainWin;
    return nullptr;
}