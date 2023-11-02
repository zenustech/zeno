/*
 Copyright (c) 2008-2022, Benoit AUTHEMAN All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the author or Destrat.io nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL AUTHOR BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

//-----------------------------------------------------------------------------
// This file is a part of the QuickQanava software library.
//
// \file	style.cpp
// \author	benoit@qanava.org
// \date	2016 02 09
//-----------------------------------------------------------------------------

// Qt headers
#include <QGuiApplication>
#include <QQuickStyle>

// QuickQanava headers
#include "../../src/QuickQanava.h"
#include "./custom.h"

using namespace qan;

int	main( int argc, char** argv )
{
    QGuiApplication app(argc, argv);
    QQuickStyle::setStyle("Material");

    qmlRegisterType< CustomRectNode >( "StyleSample", 1, 0, "CustomRectNode");
    qmlRegisterType< CustomRoundNode >( "StyleSample", 1, 0, "CustomRoundNode");
    qmlRegisterType< CustomGraph >( "StyleSample", 1, 0, "CustomGraph");
    qmlRegisterType< CustomEdge >( "StyleSample", 1, 0, "AbstractCustomEdge");

    QQmlApplicationEngine engine;
    engine.addPluginPath(QStringLiteral("../../src")); // Necessary only for development when plugin is not installed to QTDIR/qml
    QuickQanava::initialize(&engine);
    engine.load(QUrl("qrc:/style.qml"));
    return app.exec();
}

