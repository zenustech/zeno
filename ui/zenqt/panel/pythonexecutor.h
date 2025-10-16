#pragma once

#ifndef __PYTHON_EXECUTOR_H__
#define __PYTHON_EXECUTOR_H__

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QListWidget>
#include <QStackedWidget>
#include <QTextEdit>
#include <QApplication>
#include <QVBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QDialog>
#include <QPushButton>
#include <QShortcut>
#include <QDebug>
#include <QTcpServer>
#include <QTcpSocket>


class PythonExecutePane : public QWidget
{
    Q_OBJECT
public:
    PythonExecutePane(QWidget* parent = nullptr);

private slots:
    void run();

private:
    QListWidget* listWidget;
    QStackedWidget* stackedWidget;
};

class PythonAIDialog : public QDialog {
public:
    PythonAIDialog(QWidget* parent = nullptr);

private:
    QTcpSocket* m_clientSocket;
};


#endif