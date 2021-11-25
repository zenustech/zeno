/*
  This file is part of KDDockWidgets.

  SPDX-FileCopyrightText: 2019-2021 Klarälvdalens Datakonsult AB, a KDAB Group company <info@kdab.com>
  Author: Sérgio Martins <sergio.martins@kdab.com>

  SPDX-License-Identifier: GPL-2.0-only OR GPL-3.0-only

  Contact KDAB at <info@kdab.com> for commercial licensing options.
*/

/**
 * @file
 * @brief QMainWindow wrapper to enable KDDockWidgets support.
 *
 * @author Sérgio Martins \<sergio.martins@kdab.com\>
 */

#include "MainWindow.h"
#include "Config.h"
#include "FrameworkWidgetFactory.h"

#include "private/DockRegistry_p.h"
#include "private/DropAreaWithCentralFrame_p.h"
#include "private/DropArea_p.h"
#include "private/Frame_p.h"
#include "private/Logging_p.h"
#include "private/SideBar_p.h"

#include <QPainter>
#include <QScreen>
#include <QVBoxLayout>
#include <QWindow>
#include <QMenuBar>
#include <QString>

// clazy:excludeall=ctor-missing-parent-argument,missing-qobject-macro

using namespace KDDockWidgets;

namespace KDDockWidgets {
class MyCentralWidget : public QWidget
{
public:
    explicit MyCentralWidget(QWidget *parent = nullptr)
        : QWidget(parent)
    {
        setObjectName(QStringLiteral("MyCentralWidget"));
    }

    ~MyCentralWidget() override;
};
}

class MainWindow::Private
{
public:
    explicit Private(MainWindowOptions options, MainWindow *mainWindow)
        : q(mainWindow)
        , m_supportsAutoHide(Config::self().flags() & Config::Flag_AutoHideSupport)
        , m_centralWidget(new MyCentralWidget(mainWindow))
        , m_layout(new QHBoxLayout) // 1 level of indirection so we can add some margins
        , m_menubar(nullptr)
    {
        if (m_supportsAutoHide) {
            for (auto location : { SideBarLocation::North, SideBarLocation::East,
                                   SideBarLocation::West, SideBarLocation::South }) {
                m_sideBars.insert(location, Config::self().frameworkWidgetFactory()->createSideBar(location, mainWindow));
            }
        }

        if (options == MainWindowOption_HasCentralWidgetAndMenubar) {
            m_vLayout = new QVBoxLayout;
            m_vLayout->setMargin(0);
            m_vLayout->setSpacing(0);
            m_layout->setMargin(0);
            m_menubar = new QMenuBar;
            m_menubar->setMaximumHeight(26);
            //QMenu *pFile = new QMenu(tr("File"), m_menubar);
            //m_menubar->addMenu(pFile);
            m_vLayout->addWidget(m_menubar);
            m_vLayout->addLayout(m_layout);
            m_centralWidget->setLayout(m_vLayout);
        } else {
            m_centralWidget->setLayout(m_layout);
        }

        m_layout->setSpacing(0);
        updateMargins();
    }

    void updateMargins()
    {
        m_layout->setContentsMargins(q->centerWidgetMargins());
    }

    MainWindow *const q;
    const bool m_supportsAutoHide;
    QHash<SideBarLocation, SideBar *> m_sideBars;
    MyCentralWidget *const m_centralWidget;
    QHBoxLayout *const m_layout;
    QVBoxLayout *m_vLayout;
    QMenuBar *m_menubar;
};

MyCentralWidget::~MyCentralWidget()
{
}


MainWindow::MainWindow(const QString &name, MainWindowOptions options,
                       QWidget *parent, Qt::WindowFlags flags)
    : MainWindowBase(name, options, parent, flags)
    , d(new Private(options, this))
{
    if (d->m_supportsAutoHide) {
        d->m_layout->addWidget(sideBar(SideBarLocation::West));
        auto innerVLayout = new QVBoxLayout();
        innerVLayout->setSpacing(0);
        innerVLayout->setContentsMargins(0, 0, 0, 0);
        innerVLayout->addWidget(sideBar(SideBarLocation::North));
        innerVLayout->addWidget(layoutWidget());
        innerVLayout->addWidget(sideBar(SideBarLocation::South));
        d->m_layout->addLayout(innerVLayout);
        d->m_layout->addWidget(sideBar(SideBarLocation::East));
    } else {
        d->m_layout->addWidget(layoutWidget());
    }

    setCentralWidget(d->m_centralWidget);

    create();
    connect(windowHandle(), &QWindow::screenChanged, DockRegistry::self(),
            [this] {
                d->updateMargins(); // logical dpi might have changed
                Q_EMIT DockRegistry::self()->windowChangedScreen(windowHandle());
            });
}

MainWindow::~MainWindow()
{
    delete d;
}

void MainWindow::SetMenuBar(QMenuBar *menubar)
{
    d->m_menubar = menubar;
}

QMenuBar *MainWindow::MenuBar()
{
    return d->m_menubar;
}

void MainWindow::setCentralWidget(QWidget *w)
{
    QMainWindow::setCentralWidget(w);
}

SideBar *MainWindow::sideBar(SideBarLocation location) const
{
    return d->m_sideBars.value(location);
}

void MainWindow::addToCentralFrame(QWidget *pWidget)
{
    if (dropArea())
        dropArea()->addToCentralFrame(pWidget);
}

void MainWindow::resizeEvent(QResizeEvent *ev)
{
    MainWindowBase::resizeEvent(ev);
    onResized(ev); // Also call our own handler, since QtQuick doesn't have resizeEvent()
}

QMargins MainWindow::centerWidgetMargins() const
{
    const QMargins margins = { 1, 1, 1, 1 };
    return margins * logicalDpiFactor(this);
}

QRect MainWindow::centralAreaGeometry() const
{
    return centralWidget()->geometry();
}
