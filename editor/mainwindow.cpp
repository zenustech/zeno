#include "mainwindow.h"
#include "./ui_mainwindow.h"

ZENO_NAMESPACE_BEGIN

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , nodeScene(new QDMGraphicsScene)
{
    ui->setupUi(this);
    nodeScene->setName("root");
    ui->nodeView->switchScene(nodeScene.get());
    ui->treeView->setRootScene(nodeScene.get());
}

MainWindow::~MainWindow() = default;

ZENO_NAMESPACE_END
