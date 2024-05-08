// QCodeEditor
#include "ZPythonCompleter.h"
#include <QLanguage>

// Qt
#include <QStringListModel>
#include <QFile>

ZPythonCompleter::ZPythonCompleter(QObject *parent) :
    QPythonCompleter(parent)
{
    Q_INIT_RESOURCE(qcodeeditor_resources);
    QFile fl(":/languages/zpython.xml");

    if (!fl.open(QIODevice::ReadOnly))
    {
        return;
    }

    QLanguage language(&fl);

    if (!language.isLoaded())
    {
        return;
    }

    if (QStringListModel* pModel = qobject_cast<QStringListModel*>(this->model()))
    {

        auto list = pModel->stringList();
        auto keys = language.keys();
        for (auto&& key : keys)
        {
            auto names = language.names(key);
            list << names;
        }
        pModel->setStringList(list);
    }
}
