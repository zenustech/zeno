/*
    This file is part of QuickProperties2 library.

    Copyright (C) 2016  Benoit AUTHEMAN

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

//-----------------------------------------------------------------------------
// This file is a part of the QuickProperties2 library.
//
// \file	qpsContainerModelSample.cpp
// \author	benoit@qanava.org
// \date	2015 10 29
//-----------------------------------------------------------------------------

// Std headers
#include <vector>

// Qt headers
#include <QVariant>
#include <QQuickStyle>
#include <QQmlContext>

// QuickContainers headers
#include "../../src/QuickContainers.h"
#include "../../src/qcmContainerModel.h"
#include "./qcmContainerModelSample.h"

using namespace qcm;

using WeakQA = std::weak_ptr<QA>;
Q_DECLARE_METATYPE( WeakQA );

using WeakQB = std::weak_ptr<QB>;
Q_DECLARE_METATYPE( WeakQB );

using WeakQObject = std::weak_ptr<QObject>;
Q_DECLARE_METATYPE( WeakQObject );

//-----------------------------------------------------------------------------
MainView::MainView( QGuiApplication* application ) :
    QQuickView( )
{
    Q_UNUSED( application );
    QuickContainers::initialize();

    qmlRegisterType< Dummy >( "ContainerModelSample", 1, 0, "Dummy");

    auto ints = new qcm::ContainerModel< QVector, int >( this );
    ints->append(42);
    ints->append(43);
    ints->append(44);
    ints->append(45);
    rootContext( )->setContextProperty( "ints", ints );

    auto dummies = new qcm::ContainerModel< QVector, Dummy* >( this );
    dummies->append( new Dummy{"First", 42.} );
    dummies->append( new Dummy{"Second", 43.} );
    rootContext( )->setContextProperty( "dummies", dummies );

/*
    using DummyPtr = QPointer<Dummy>;
    auto qptrDummies = new qcm::ContainerModel< QVector, QPointer<Dummy> >( this );
    qptrDummies->append( DummyPtr{new Dummy{"First", 42.}} );
    qptrDummies->append( DummyPtr{new Dummy{"Second", 43.}} );
    rootContext( )->setContextProperty( "qptrDummies", qptrDummies );
*/

    auto dummies1 = new qcm::ContainerModel< QVector, Dummy* >( this );
    dummies1->append( new Dummy{"First", 42.} );
    dummies1->append( new Dummy{"Second", 43.} );
    rootContext( )->setContextProperty( "dummies1", dummies1 );

    auto dummies2 = new qcm::ContainerModel< QVector, Dummy* >( this );
    dummies2->append( new Dummy{"Third", 44.} );
    dummies2->append( new Dummy{"Fourth", 45.} );
    rootContext( )->setContextProperty( "dummies2", dummies2 );

    // Used in heterogeneous model composer
    // m1 is a vector of std::weak_ptr<QA> smart pointers
    // m2 and m3 are vectors of std::weak_ptr<QB> smart pointers
    // QA and QB are QObjects.
    // We are composing m1 and m3 in an heterogeneous model
    // of std::weak_ptr<QObject> since QA and QB have a common base
    // class.
    // m2 is swapped with m3 at runtime...
    using SharedQAs = QVector<std::shared_ptr<QA>>;
    auto sharedQAs = new SharedQAs{};   // Create a vector of shared QAs and QBs to avoid their destruction at the end
                                        // of the block.
    using WeakQAs = qcm::ContainerModel< QVector, WeakQA >;
    auto m1{new WeakQAs{}};
    auto m1o1Ptr{std::make_shared<QA>(42)}; auto m1o1{WeakQA{m1o1Ptr}};
    auto m1o2Ptr{std::make_shared<QA>(43)}; auto m1o2{WeakQA{m1o2Ptr}};
    auto m1o3Ptr{std::make_shared<QA>(44)}; auto m1o3{WeakQA{m1o3Ptr}};
    sharedQAs->append(m1o1Ptr); sharedQAs->append(m1o2Ptr); sharedQAs->append(m1o3Ptr);
    m1->append( m1o1 ); m1->append( m1o2 ); m1->append( m1o3 );

    using WeakQBs = qcm::ContainerModel< QVector, WeakQB >;
    using SharedQBs = QVector<std::shared_ptr<QB>>;
    auto sharedQBs = new SharedQBs{};
    auto m2{new WeakQBs{}};
    auto m2o1Ptr{std::make_shared<QB>(45)}; auto m2o1{WeakQB{m2o1Ptr}};
    auto m2o2Ptr{std::make_shared<QB>(46)}; auto m2o2{WeakQB{m2o2Ptr}};
    auto m2o3Ptr{std::make_shared<QB>(47)}; auto m2o3{WeakQB{m2o3Ptr}};
    sharedQBs->append(m2o1Ptr); sharedQBs->append(m2o2Ptr); sharedQBs->append(m2o3Ptr);
    m2->append( m2o1 ); m2->append( m2o2 ); m2->append( m2o3 );

    auto m3{new WeakQBs{}};
    auto m3o1Ptr{std::make_shared<QB>(48)}; auto m3o1{WeakQB{m3o1Ptr}};
    auto m3o2Ptr{std::make_shared<QB>(49)}; auto m3o2{WeakQB{m3o2Ptr}};
    auto m3o3Ptr{std::make_shared<QB>(50)}; auto m3o3{WeakQB{m3o3Ptr}};
    sharedQBs->append(m3o1Ptr); sharedQBs->append(m3o2Ptr); sharedQBs->append(m3o3Ptr);
    m3->append( m3o1 ); m3->append( m3o2 ); m3->append( m3o3 );

    rootContext( )->setContextProperty( "m1", m1 );
    rootContext( )->setContextProperty( "m2", m2 );
    rootContext( )->setContextProperty( "m3", m3 );

    setSource( QUrl( "qrc:/main.qml" ) );

    // It's a sample, let's leak everything for a time :)
}
//-----------------------------------------------------------------------------

int	main( int argc, char** argv )
{
    QGuiApplication app(argc, argv);
    QQuickStyle::setStyle("Material");
    MainView mainView( &app );
    mainView.setResizeMode( QQuickView::SizeRootObjectToView );
    mainView.resize( 800, 800 );
    mainView.show( );

    return app.exec( );
}
