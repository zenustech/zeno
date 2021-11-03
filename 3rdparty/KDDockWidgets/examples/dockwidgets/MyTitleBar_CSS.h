/*
  This file is part of KDDockWidgets.

  SPDX-FileCopyrightText: 2019-2021 Klarälvdalens Datakonsult AB, a KDAB Group company <info@kdab.com>
  Author: Sérgio Martins <sergio.martins@kdab.com>

  SPDX-License-Identifier: GPL-2.0-only OR GPL-3.0-only

  Contact KDAB at <info@kdab.com> for commercial licensing options.
*/

#ifndef EXAMPLETITLEBAR_CSS_H
#define EXAMPLETITLEBAR_CSS_H

#pragma once

#include <kddockwidgets/private/widgets/TitleBarWidget_p.h>

/**
 * @brief Shows how to implement a custom titlebar which uses "Qt StyleSheets".
 *
 * Derive from KDDockWidgets::DefaultWidgetFactory and override the two createTitleBar() methods.
 *
 * To try it out, modify examples/dockwidgets/MyFrameworkWidgetFactory.cpp to return a MyTitleBar_CSS instance.
 * Run the example with: ./bin/kddockwidgets_example -p
 *
 * WARNINGS:
 *   - Qt StyleSheets are not recommended for new applications. Often you are able to style 90% of
 *   the application but then hit a road block. QStyle is much more powerful and flexible.
 *   - The Qt maintainers have manifested intention to deprecated stylesheets.
 *   - Stylesheets are supported for built-in QWidgets (QPushButton, QComboBox, etc.), any widget
 *   that's not in Qt needs to be crafted by the user, that includes, for example, having to paint
 *   your background manually. KDDockWidget::TitleBarWidget does this for your convenience though.
 *   - Qt stylesheets don't react to property changes (known old bug in Qt), for example:
 *     QLineEdit[readOnly="true"] { color: gray }
 *     this won't trigger when readOnly changes to false, you need to set/unset. This is QTBUG-51236
 *   - KDDockWidgets::TitleBarWidget::isFocused is a property, there for needs to workaround the
 *   above bug by unsetting the sheet and setting it again.
 */
class MyTitleBar_CSS : public KDDockWidgets::TitleBarWidget
{
public:
    explicit MyTitleBar_CSS(KDDockWidgets::Frame *frame)
        : KDDockWidgets::TitleBarWidget(frame)
    {
        init();
    }

    explicit MyTitleBar_CSS(KDDockWidgets::FloatingWindow *fw)
        : KDDockWidgets::TitleBarWidget(fw)
    {
        init();
    }

    ~MyTitleBar_CSS() override;

    void initStyleSheet()
    {
        // Or use qApp->setStyleSheet(), as you prefer
        setStyleSheet(QStringLiteral("KDDockWidgets--TitleBarWidget {"
                                     "background: blue"
                                     "}"
                                     "KDDockWidgets--TitleBarWidget:hover {"
                                     "background: red"
                                     "}"
                                     "KDDockWidgets--TitleBarWidget[isFocused=true] {"
                                     "background: green"
                                     "}"));
    }

    void init()
    {
        initStyleSheet();
        connect(this, &KDDockWidgets::TitleBar::isFocusedChanged, this, [this] {
            // Workaround QTBUG-51236, this makes the [isFocused=true] syntax useful
            setStyleSheet(QString());
            initStyleSheet();
        });
    }
};

MyTitleBar_CSS::~MyTitleBar_CSS()
{
}

#endif
