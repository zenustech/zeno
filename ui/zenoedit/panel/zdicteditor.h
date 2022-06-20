#ifndef __ZDICT_EDITOR_H__
#define __ZDICT_EDITOR_H__

namespace Ui {
    class DictEditor;
}

#include <QtWidgets>

class DictModel;

class ZDictEditor : public QWidget
{
    Q_OBJECT
public:
    ZDictEditor(DictModel* pModel, QWidget* parent = nullptr);

private slots:
    void onAddClicked();

private:
    Ui::DictEditor* m_ui;
    QStandardItemModel* m_model;
};


#endif