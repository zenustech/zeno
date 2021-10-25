#include "mainwindow.h"
#include "./ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(std::make_unique<Ui::MainWindow>())
    , nodeScene(std::make_unique<QDMGraphicsScene>())
{
    ui->setupUi(this);
    ui->nodeView->setScene(nodeScene.get());
}

MainWindow::~MainWindow() = default;
