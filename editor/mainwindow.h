#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "qdmgraphicsscene.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

ZENO_NAMESPACE_BEGIN

class MainWindow : public QMainWindow
{
    Q_OBJECT

    std::unique_ptr<QDMGraphicsScene> nodeScene;
    std::unique_ptr<Ui::MainWindow> ui;

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
};

ZENO_NAMESPACE_END

#endif // MAINWINDOW_H
