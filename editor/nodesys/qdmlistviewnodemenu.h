#ifndef QDMLISTVIEWNODEMENU_H
#define QDMLISTVIEWNODEMENU_H

#include <zeno/common.h>
#include <QListView>

ZENO_NAMESPACE_BEGIN

class QDMListViewNodeMenu : public QListView
{
    Q_OBJECT

public:
    explicit QDMListViewNodeMenu(QWidget *parent = nullptr);
    ~QDMListViewNodeMenu();

signals:
    void entryClicked(QString name);
};

ZENO_NAMESPACE_END

#endif // QDMLISTVIEWNODEMENU_H
