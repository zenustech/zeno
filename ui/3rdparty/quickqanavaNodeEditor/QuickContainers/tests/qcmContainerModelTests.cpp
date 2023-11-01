/*
 Copyright (c) 2008-2017, Benoit AUTHEMAN All rights reserved.

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
// This file is a part of QuickContainers library.
//
// \file	qpsContainerModelTests.cpp
// \author	benoit@qanava.org
// \date	2016 11 25
//-----------------------------------------------------------------------------

// Qt headers
#include <memory>

// Qt headers
#include <QSignalSpy>

// QuickContainers headers
#include "./qcmTests.h"
#include "./qcmContainerModelTests.h"

//-----------------------------------------------------------------------------
// Static tag dispatching test
//-----------------------------------------------------------------------------
TEST(qpsContainerModel, staticDispatch)
{
    //std::cerr << "qcm::ItemDispatcher<int>::type=" << qcm::ItemDispatcherBase::debug_type< qcm::ItemDispatcher<int>::type >() << std::endl;
    bool isNonPtr{ std::is_same< qcm::ItemDispatcher<int>::type, qcm::ItemDispatcherBase::non_ptr_type>::value };
    EXPECT_TRUE( isNonPtr );

    //std::cerr << "qcm::ItemDispatcher<int*>::type=" << qcm::ItemDispatcherBase::debug_type< qcm::ItemDispatcher<int*>::type >() << std::endl;
    bool isPtr{ std::is_same< qcm::ItemDispatcher<int*>::type, qcm::ItemDispatcherBase::ptr_type>::value };
    EXPECT_TRUE( isPtr );

    //std::cerr << "qcm::ItemDispatcher<QObject>::type=" << qcm::ItemDispatcherBase::debug_type< qcm::ItemDispatcher<QObject>::type >() << std::endl;
    bool isUnsupported{ std::is_same< qcm::ItemDispatcher<QObject>::type, qcm::ItemDispatcherBase::unsupported_type>::value };
    EXPECT_TRUE( isUnsupported );

    //std::cerr << "qcm::ItemDispatcher<QObject*>::type=" << qcm::ItemDispatcherBase::debug_type< qcm::ItemDispatcher<QObject*>::type >() << std::endl;
    bool isQObjectPtr{ std::is_same< qcm::ItemDispatcher<QObject*>::type, qcm::ItemDispatcherBase::ptr_qobject_type>::value };
    EXPECT_TRUE( isQObjectPtr );

    //std::cerr << "qcm::ItemDispatcher<QPointer<QObject>>::type=" << qcm::ItemDispatcherBase::debug_type< qcm::ItemDispatcher<QPointer<QObject>>::type >() << std::endl;
    bool isQPointer{ std::is_same< qcm::ItemDispatcher<QPointer<QObject>>::type, qcm::ItemDispatcherBase::q_ptr_type>::value };
    EXPECT_TRUE( isQPointer );

    //std::cerr << "qcm::ItemDispatcher<std::shared_ptr<int>>::type=" << qcm::ItemDispatcherBase::debug_type< qcm::ItemDispatcher<std::shared_ptr<int>>::type >() << std::endl;
    bool isSharedPtr{ std::is_same< qcm::ItemDispatcher<std::shared_ptr<int>>::type, qcm::ItemDispatcherBase::shared_ptr_type>::value };
    EXPECT_TRUE( isSharedPtr );

    //std::cerr << "qcm::ItemDispatcher<std::shared_ptr<QObject>>::type=" << qcm::ItemDispatcherBase::debug_type< qcm::ItemDispatcher<std::shared_ptr<QObject>>::type >() << std::endl;
    bool isSharedQObjectPtr{ std::is_same< qcm::ItemDispatcher<std::shared_ptr<QObject>>::type, qcm::ItemDispatcherBase::shared_ptr_qobject_type>::value };
    EXPECT_TRUE( isSharedQObjectPtr );

    //std::cerr << "qcm::ItemDispatcher<std::weak_ptr<int>>::type=" << qcm::ItemDispatcherBase::debug_type< qcm::ItemDispatcher<std::weak_ptr<int>>::type >() << std::endl;
    bool isWeakPtr{ std::is_same< qcm::ItemDispatcher<std::weak_ptr<int>>::type, qcm::ItemDispatcherBase::weak_ptr_type>::value };
    EXPECT_TRUE( isWeakPtr );

    //std::cerr << "qcm::ItemDispatcher<std::weak_ptr<QObject>>::type=" << qcm::ItemDispatcherBase::debug_type< qcm::ItemDispatcher<std::weak_ptr<QObject>>::type >() << std::endl;
    bool isWeakQObjectPtr{ std::is_same< qcm::ItemDispatcher<std::weak_ptr<QObject>>::type, qcm::ItemDispatcherBase::weak_ptr_qobject_type>::value };
    EXPECT_TRUE( isWeakQObjectPtr );
}


//-----------------------------------------------------------------------------
// qcm::Adapter tests
//-----------------------------------------------------------------------------
template <class Adapter, class C>
void    testAdapterSequenceInt()
{
    C ints;

    // Adapter<>::reserve()
    Adapter::reserve(ints, 10);

    // Adapter<>::insert(&)
    Adapter::insert(ints, 42);
    EXPECT_EQ(1, ints.size());

    // Adapter<>::insert(&&)
    int i{43};
    Adapter::insert(ints, std::move(i));
    EXPECT_EQ(2, ints.size());

    // Adapter<>::indexOf() / Adapter<>::contains()
    EXPECT_EQ(1, Adapter::indexOf(ints, 43));
    EXPECT_TRUE(Adapter::contains(ints, 43));
    EXPECT_FALSE(Adapter::contains(ints, 4343));

    // Adapter<>::append(&)
    Adapter::append(ints, 72);
    EXPECT_EQ(3, ints.size());

    // Adapter<>::append(&&)
    int i2{43};
    Adapter::append(ints, std::move(i2));
    EXPECT_EQ(4, ints.size());  // 43 already inserted
    int i3{44};
    Adapter::append(ints, std::move(i3));
    EXPECT_EQ(5, ints.size());

    // Adapter<>::remove()
    Adapter::remove(ints, 0);       // Remove 42
    EXPECT_EQ(4, ints.size());
    EXPECT_EQ(0, Adapter::indexOf(ints, 43));
    EXPECT_EQ(-1, Adapter::indexOf(ints, 4242));

    // Adapter<>::removeAll()
    Adapter::insert(ints, 42);
    Adapter::insert(ints, 43);
    Adapter::insert(ints, 42);

    // Can't have the removed item count with std::vector
    const auto originalSize = ints.size();
    // Can't test removeAll, with std it _always_ return -1
    Adapter::removeAll(ints, 42);
    //EXPECT_EQ(-1, Adapter::removeAll(ints, 42));
    EXPECT_EQ(originalSize - 2, ints.size());
}

TEST( qcmAdapter, qVectorInt )
{
    using Adapter = qcm::adapter<QVector, int>;
    testAdapterSequenceInt<Adapter, QVector<int>>();
}

TEST( qcmAdapter, qListInt )
{
    using Adapter = qcm::adapter<QList, int>;
    testAdapterSequenceInt<Adapter, QList<int>>();
}

TEST( qcmAdapter, stdVectorInt )
{
    using Adapter = qcm::adapter<std::vector, int>;
    testAdapterSequenceInt<Adapter, std::vector<int>>();
}

template <class Adapter, class C>
void    testAssociativeSequenceInt()
{
    C ints;

    // Adapter<>::reserve()
    Adapter::reserve(ints, 10);

    // Adapter<>::insert(&)
    Adapter::insert(ints, 42);
    EXPECT_EQ(1, ints.size());

    // Adapter<>::insert(&&)
    int i{43};
    Adapter::insert(ints, std::move(i));
    EXPECT_EQ(2, ints.size());

    // Adapter<>::indexOf() / Adapter<>::contains()
        // Testing indexOf() has no sense with associative containers (QSet is an hashtable)
    EXPECT_NE(Adapter::indexOf(ints, 42), Adapter::indexOf(ints, 43));
    EXPECT_EQ(-1, Adapter::indexOf(ints, 4242));
    EXPECT_TRUE(Adapter::contains(ints, 42));
    EXPECT_FALSE(Adapter::contains(ints, 4242));

    // Adapter<>::append(&)
    Adapter::append(ints, 72);
    EXPECT_EQ(3, ints.size());

    // Adapter<>::append(&&)
    int i2{43};
    Adapter::append(ints, std::move(i2));
    EXPECT_EQ(3, ints.size());  // 43 already inserted
    int i3{44};
    Adapter::append(ints, std::move(i3));
    EXPECT_EQ(4, ints.size());

    // Adapter<>::remove()
    Adapter::remove(ints, 0);       // Remove "a value"!
    EXPECT_EQ(3, ints.size());

    // Adapter<>::removeAll()
    Adapter::insert(ints, 42);
    Adapter::insert(ints, 43);
    Adapter::insert(ints, 42);

    // Can't have the removed item count with std::vector
    const auto originalSize = ints.size();
    // Can't test removeAll, with std it _always_ return -1
    Adapter::removeAll(ints, 42);
    EXPECT_EQ(originalSize - 1, ints.size());
}

TEST( qcmAdapter, qSetInt )
{
    using Adapter = qcm::adapter<QSet, int>;
    testAssociativeSequenceInt<Adapter, QSet<int>>();
}

//-----------------------------------------------------------------------------
// qcm::Container QVector POD tests
//-----------------------------------------------------------------------------
TEST(qpsContainer, qVectorPod)
{
    using Ints = qcm::Container< QVector, int >;
    {
        Ints ints;
        EXPECT_EQ( ints.model()->rowCount(), 0 );
        EXPECT_EQ( ints.model()->getLength(), 0 );
    }

    {   // adapter::reserve()
        Ints ints;
        ints.reserve(10);
        EXPECT_EQ( ints.model()->rowCount(), 0 );
        EXPECT_EQ( ints.model()->getLength(), 0 );
    }

    {   // adapter::append()
        Ints ints;
        EXPECT_EQ( ints.model()->getLength(), 0 );
        ints.append( 42 );
        ints.append( 43 );
        EXPECT_EQ( ints.model()->getLength(), 2 );
    }

    {   // adapter::remove()
        Ints ints;
        ints.append(42);
        ints.append(43);
        EXPECT_EQ( ints.model()->getLength(), 2 );
        ints.removeAll(43);
        EXPECT_EQ( ints.model()->getLength(), 1 );
        ints.removeAll(42);
        EXPECT_EQ( ints.model()->getLength(), 0 );
    }
}

//-----------------------------------------------------------------------------
// qcm::Container std::vector POD tests
//-----------------------------------------------------------------------------
TEST(qpsContainer, stdVectorPod)
{
    using Ints = qcm::Container<std::vector, int >;
    {
        Ints ints;
        EXPECT_EQ( ints.model()->rowCount(), 0 );
        EXPECT_EQ( ints.model()->getLength(), 0 );
    }

    {   // adapter::reserve()
        Ints ints;
        ints.reserve(10);
        EXPECT_EQ( ints.model()->rowCount(), 0 );
        EXPECT_EQ( ints.model()->getLength(), 0 );
    }

    {   // adapter::append()
        Ints ints;
        EXPECT_EQ( ints.model()->getLength(), 0 );
        ints.append( 42 );
        ints.append( 43 );
        EXPECT_EQ( ints.model()->getLength(), 2 );
    }

    {   // adapter::remove()
        Ints ints;
        ints.append(42);
        ints.append(43);
        EXPECT_EQ( ints.model()->getLength(), 2 );
        ints.removeAll(43);
        EXPECT_EQ( ints.model()->getLength(), 1 );
        ints.removeAll(42);
        EXPECT_EQ( ints.model()->getLength(), 0 );
    }
}

TEST(qpsContainer, qVectorQObjectPtrEmpty)
{
    using QObjects = qcm::Container<QVector, QObject*>;
    QObjects objects;
    EXPECT_EQ( objects.model()->rowCount(), 0 );
    EXPECT_EQ( objects.model()->getLength(), 0 );
    EXPECT_EQ( objects.size(), 0 );

    objects.append( new QObject{nullptr} );
    objects.append( new QObject{nullptr} );
    EXPECT_EQ( objects.model()->rowCount(), 2 );
    EXPECT_EQ( objects.model()->getLength(), 2 );
    EXPECT_EQ( objects.size(), 2 );

    objects.clear();
    QObject* o = new QObject();
    o->setObjectName( "Hola" );

    // ContainerModel::append()
    objects.append( o );
    objects.append( new QObject() );
    objects.append( new QObject() );
    objects.append( nullptr );  // Shouldn't throw
    EXPECT_EQ( objects.size(), 3 );
    EXPECT_EQ( objects.at(0)->objectName(), "Hola" );

    // ContainerModel::remove()
    objects.removeAll( o );
    objects.removeAll( nullptr );
    EXPECT_EQ( objects.size(), 2 );

    // ContainerModel::insert()
    objects.insert( new QObject(), 0 );
    EXPECT_EQ( objects.size(), 3 );

    // ContainerModel::indexOf()
    QObject* o2 = new QObject();
    o2->setObjectName( "42" );
    objects.insert( o2, 1 );
    EXPECT_EQ( objects.indexOf( o2 ), 1 );
}

//-----------------------------------------------------------------------------
// Container model tests
//-----------------------------------------------------------------------------

TEST(qpsContainerModel, qVectorQObjectEmpty)
{
    // Expect invalid return for invalid input arguments
    using QObjects = qcm::Container< QVector, QObject* >;
    QObjects objects;
    auto model = objects.getModel();
    ASSERT_TRUE( model != nullptr );
    //ASSERT_TRUE( ( model->itemAt(-1) == QVariant{} ) );   // EXPECT -1, argument is invalid
    //ASSERT_TRUE( ( model->itemAt(0) == QVariant{} ) );    // EXPECT invalid return QVariant, index is invalid

    ASSERT_TRUE( ( model->at(-1) == nullptr ) );   // EXPECT nullptr, argument is invalid
    ASSERT_TRUE( ( model->at(0) == nullptr ) );    // EXPECT nullptr, index is invalid
}

TEST(qpsContainerModel, qVectorQObject)
{
    using QObjects = qcm::Container< QVector, QObject* >;
    QObjects objects;

    QObject* o1{new QObject()};
    QObject* o2{new QObject()};

    auto model = objects.getModel();
    ASSERT_TRUE( model != nullptr );

    // qcm::ContainerModel::append()
    ASSERT_EQ( 0, model->getLength() );
    model->append( o1 );
    ASSERT_EQ( 1, model->getLength() );
    model->append( new QObject() );
    model->append( o2 );
    ASSERT_EQ( 3, model->getLength() );

    // qcm::ContainerModel::at()
    ASSERT_TRUE( ( model->at(0) != nullptr ) );
    EXPECT_TRUE( ( model->at(0) == o1 ) );
    EXPECT_TRUE( ( model->at(2) == o2 ) );
    ASSERT_TRUE( ( model->at(3) == nullptr ) );  // Overflow

    // qcm::ContainerModel::at()
    QObject* o3{new QObject()};
    ASSERT_EQ( -1, model->indexOf(nullptr) );   // Invalid input
    ASSERT_EQ( -1, model->indexOf(o3) );        // Invalid input, o3 is not inserted
    EXPECT_EQ( 0, model->indexOf(o1) );
    EXPECT_EQ( 2, model->indexOf(o2) );

    // qcm::ContainerModel::remove()
    model->remove(nullptr);                 // Invalid input
    model->remove(o3);                      // Invalid input (o3 is not inserted)
    model->remove(o1);
    ASSERT_EQ( 2, model->getLength() );

    // qcm::ContainerModel::contains()
    ASSERT_FALSE( model->contains(nullptr) );    // Invalid input
    ASSERT_FALSE( model->contains(o3) );         // Invalid input o3 is not inserted
    ASSERT_FALSE( model->contains(o1) );         // o1 has been removce
    ASSERT_TRUE( model->contains(o2) );

    // qcm::ContainerModel::clear()
    model->clear();
    ASSERT_TRUE( model->at(0) == nullptr );
    ASSERT_EQ( 0, model->getLength() );
}

TEST(qpsContainerModel, qListQObject)
{
    using QObjects = qcm::Container< QList, QObject* >;
    QObjects objects;

    QObject* o1{new QObject()};
    QObject* o2{new QObject()};

    auto model = objects.getModel();
    ASSERT_TRUE( model != nullptr );

    // qcm::ContainerModel::append()
    ASSERT_EQ( 0, model->getLength() );
    model->append( o1 );
    ASSERT_EQ( 1, model->getLength() );
    model->append( new QObject() );
    model->append( o2 );
    ASSERT_EQ( 3, model->getLength() );

    // qcm::ContainerModel::at()
    ASSERT_TRUE( ( model->at(0) != nullptr ) );
    EXPECT_TRUE( ( model->at(0) == o1 ) );
    EXPECT_TRUE( ( model->at(2) == o2 ) );
    ASSERT_TRUE( ( model->at(3) == nullptr ) );  // Overflow

    // qcm::ContainerModel::at()
    QObject* o3{new QObject()};
    ASSERT_EQ( -1, model->indexOf(nullptr) );   // Invalid input
    ASSERT_EQ( -1, model->indexOf(o3) );        // Invalid input, o3 is not inserted
    EXPECT_EQ( 0, model->indexOf(o1) );
    EXPECT_EQ( 2, model->indexOf(o2) );

    // qcm::ContainerModel::remove()
    model->remove(nullptr);                 // Invalid input
    model->remove(o3);                      // Invalid input (o3 is not inserted)
    model->remove(o1);
    ASSERT_EQ( 2, model->getLength() );

    // qcm::ContainerModel::contains()
    ASSERT_FALSE( model->contains(nullptr) );    // Invalid input
    ASSERT_FALSE( model->contains(o3) );         // Invalid input o3 is not inserted
    ASSERT_FALSE( model->contains(o1) );         // o1 has been removce
    ASSERT_TRUE( model->contains(o2) );

    // qcm::ContainerModel::clear()
    model->clear();
    ASSERT_TRUE( model->at(0) == nullptr );
    ASSERT_EQ( 0, model->getLength() );
}

TEST(qpsContainerModel, qSetQObject)
{
    using QObjects = qcm::Container< QList, QObject* >;
    QObjects objects;

    QObject* o1{new QObject()};
    QObject* o2{new QObject()};

    auto model = objects.getModel();
    ASSERT_TRUE( model != nullptr );

    // qcm::ContainerModel::append()
    ASSERT_EQ( 0, model->getLength() );
    model->append( o1 );
    ASSERT_EQ( 1, model->getLength() );
    model->append( new QObject() );
    model->append( o2 );
    ASSERT_EQ( 3, model->getLength() );
    //model->append( o2 );                    // Inserting same value should not increment length with QSet
    //ASSERT_EQ( 3, model->getLength() );

    // qcm::ContainerModel::at() / indexOf()
    QObject* o3{new QObject()};
    ASSERT_EQ( -1, model->indexOf(nullptr) );   // Invalid input
    ASSERT_EQ( -1, model->indexOf(o3) );        // Invalid input, o3 is not inserted
    EXPECT_TRUE( ( model->at(0) == o1 ) );
    EXPECT_TRUE( ( model->at(model->indexOf(o1)) == o1 ) );
    EXPECT_TRUE( ( model->at(model->indexOf(o2)) == o2 ) );

    // qcm::ContainerModel::remove()
    model->remove(nullptr);                 // Invalid input
    model->remove(o3);                      // Invalid input (o3 is not inserted)
    model->remove(o1);
    ASSERT_EQ( 2, model->getLength() );

    // qcm::ContainerModel::contains()
    ASSERT_FALSE( model->contains(nullptr) );    // Invalid input
    ASSERT_FALSE( model->contains(o3) );         // Invalid input o3 is not inserted
    ASSERT_FALSE( model->contains(o1) );         // o1 has been removce
    ASSERT_TRUE( model->contains(o2) );

    // qcm::ContainerModel::clear()
    model->clear();
    ASSERT_TRUE( model->at(0) == nullptr );
    ASSERT_EQ( 0, model->getLength() );
}

TEST(qpsContainerModel, stdVectorQObject)
{
    using QObjects = qcm::Container< std::vector, QObject* >;
    QObjects objects;

    QObject* o1{new QObject()};
    QObject* o2{new QObject()};

    auto model = objects.getModel();
    ASSERT_TRUE( model != nullptr );

    // qcm::ContainerModel::append()
    ASSERT_EQ( 0, model->getLength() );
    model->append( o1 );
    ASSERT_EQ( 1, model->getLength() );
    model->append( new QObject() );
    model->append( o2 );
    ASSERT_EQ( 3, model->getLength() );

    // qcm::ContainerModel::at()
    ASSERT_TRUE( ( model->at(0) != nullptr ) );
    EXPECT_TRUE( ( model->at(0) == o1 ) );
    EXPECT_TRUE( ( model->at(2) == o2 ) );
    ASSERT_TRUE( ( model->at(3) == nullptr ) );  // Overflow

    // qcm::ContainerModel::at()
    QObject* o3{new QObject()};
    ASSERT_EQ( -1, model->indexOf(nullptr) );   // Invalid input
    ASSERT_EQ( -1, model->indexOf(o3) );        // Invalid input, o3 is not inserted
    EXPECT_EQ( 0, model->indexOf(o1) );
    EXPECT_EQ( 2, model->indexOf(o2) );

    // qcm::ContainerModel::remove()
    model->remove(nullptr);                 // Invalid input
    model->remove(o3);                      // Invalid input (o3 is not inserted)
    model->remove(o1);
    ASSERT_EQ( 2, model->getLength() );

    // qcm::ContainerModel::contains()
    ASSERT_FALSE( model->contains(nullptr) );    // Invalid input
    ASSERT_FALSE( model->contains(o3) );         // Invalid input o3 is not inserted
    ASSERT_FALSE( model->contains(o1) );         // o1 has been removce
    ASSERT_TRUE( model->contains(o2) );

    // qcm::ContainerModel::clear()
    model->clear();
    ASSERT_TRUE( model->at(0) == nullptr );
    ASSERT_EQ( 0, model->getLength() );
}

template <class C>
void    testContainerModelStdSharedQObject(C& container)
{
    auto model = container.getModel();
    ASSERT_TRUE( model != nullptr );

    auto o1{std::make_shared<QObject>()};
    auto o2{std::make_shared<QObject>()};

    // qcm::ContainerModel::append()
    ASSERT_EQ( 0, model->getLength() );
    container.append( o1 );
    ASSERT_EQ( 1, model->getLength() );
    container.append(std::make_shared<QObject>());
    container.append( o2 );
    ASSERT_EQ( 3, model->getLength() );

    // qcm::ContainerModel::at()
    ASSERT_TRUE( ( model->at(0) != nullptr ) );
    EXPECT_TRUE( ( model->at(0) == o1.get() ) );
    EXPECT_TRUE( ( model->at(2) == o2.get() ) );
    ASSERT_TRUE( ( model->at(3) == nullptr ) );  // Overflow

    // qcm::ContainerModel::at()
    QObject* o3{new QObject()};
    ASSERT_EQ( -1, model->indexOf(nullptr) );   // Invalid input
    ASSERT_EQ( -1, model->indexOf(o3) );        // Invalid input, o3 is not inserted
    EXPECT_EQ( 0, model->indexOf(o1.get()) );
    EXPECT_EQ( 2, model->indexOf(o2.get()) );

    // qcm::ContainerModel::remove()
    model->remove(nullptr);                 // Invalid input
    model->remove(o3);                      // Invalid input (o3 is not inserted)
    model->remove(o1.get());
    ASSERT_EQ( 2, model->getLength() );

    // qcm::ContainerModel::contains()
    ASSERT_FALSE( model->contains(nullptr) );    // Invalid input
    ASSERT_FALSE( model->contains(o3) );         // Invalid input o3 is not inserted
    ASSERT_FALSE( model->contains(o1.get()) );         // o1 has been removce
    ASSERT_TRUE( model->contains(o2.get()) );

    // qcm::ContainerModel::clear()
    model->clear();
    ASSERT_TRUE( model->at(0) == nullptr );
    ASSERT_EQ( 0, model->getLength() );
}

TEST(qpsContainerModel, qVectorStdSharedQObject)
{
    using SharedQObjects = qcm::Container<QVector, std::shared_ptr<QObject>>;
    SharedQObjects objects;
    testContainerModelStdSharedQObject<SharedQObjects>(objects);
}


TEST(qpsContainerModel, qListStdSharedQObject)
{
    using SharedQObjects = qcm::Container<QList, std::shared_ptr<QObject>>;
    SharedQObjects objects;
    testContainerModelStdSharedQObject<SharedQObjects>(objects);
}

TEST(qpsContainerModel, stdVectorStdSharedQObject)
{
    using SharedQObjects = qcm::Container<std::vector, std::shared_ptr<QObject>>;
    SharedQObjects objects;
    testContainerModelStdSharedQObject<SharedQObjects>(objects);
}

template <class C>
void    testContainerModelStdWeakQObject(C& container)
{
    auto model = container.getModel();
    ASSERT_TRUE( model != nullptr );

    using T = typename C::Item_type;

    auto so1{std::make_shared<QObject>()};
    auto so2{std::make_shared<QObject>()};
    auto o1 = T{so1};
    auto o2 = T{so2};

    // qcm::ContainerModel::append()
    auto so3 = std::make_shared<QObject>();
    ASSERT_EQ( 0, model->getLength() );
    container.append( o1 );
    ASSERT_EQ( 1, model->getLength() );
    container.append(so3);
    container.append( o2 );
    ASSERT_EQ( 3, model->getLength() );

    // qcm::ContainerModel::at()
    ASSERT_TRUE( ( model->at(0) != nullptr ) );
    EXPECT_TRUE( ( model->at(0) == o1.lock().get() ) );
    EXPECT_TRUE( ( model->at(2) == o2.lock().get() ) );
    ASSERT_TRUE( ( model->at(3) == nullptr ) );  // Overflow

    // qcm::ContainerModel::at()
    QObject* o3{new QObject()};
    ASSERT_EQ( -1, model->indexOf(nullptr) );   // Invalid input
    ASSERT_EQ( -1, model->indexOf(o3) );        // Invalid input, o3 is not inserted
    EXPECT_EQ( 0, model->indexOf(o1.lock().get()) );
    EXPECT_EQ( 2, model->indexOf(o2.lock().get()) );

    // qcm::ContainerModel::remove()
    model->remove(nullptr);                 // Invalid input
    model->remove(o3);                      // Invalid input (o3 is not inserted)
    model->remove(o1.lock().get());
    ASSERT_EQ( 2, model->getLength() );

    // qcm::ContainerModel::contains()
    ASSERT_FALSE( model->contains(nullptr) );    // Invalid input
    ASSERT_FALSE( model->contains(o3) );         // Invalid input o3 is not inserted
    ASSERT_FALSE( model->contains(o1.lock().get()) );         // o1 has been removce
    ASSERT_TRUE( model->contains(o2.lock().get()) );

    // qcm::ContainerModel::clear()
    model->clear();
    ASSERT_TRUE( model->at(0) == nullptr );
    ASSERT_EQ( 0, model->getLength() );
}

TEST(qpsContainerModel, qVectorStdWeakQObject)
{
    using WeakQObjects = qcm::Container<QVector, std::weak_ptr<QObject>>;
    WeakQObjects objects;
    testContainerModelStdWeakQObject<WeakQObjects>(objects);
}
TEST(qpsContainerModel, stdVectorStdWeakQObject)
{
    using WeakQObjects = qcm::Container<std::vector, std::weak_ptr<QObject>>;
    WeakQObjects objects;
    testContainerModelStdWeakQObject<WeakQObjects>(objects);
}

//-----------------------------------------------------------------------------
// Container model item display role test
//-----------------------------------------------------------------------------
template <class Container>
void    testItemDisplayRole(Container& container)
{
    auto& model = *container.model();
    DataChangedSignalSpy spy(model);
    ASSERT_EQ(0, spy.count);

    // Content is not monitored for changes _before_ beeing displayed,
    // simulate a view access to QAbstractItemModel::data()
    model.data(model.index(0,0), qcm::ContainerModel::ItemLabelRole);
    model.data(model.index(1,0), qcm::ContainerModel::ItemLabelRole);
    ASSERT_EQ(0, spy.count);

    // Now modify container model qobject properties used as display role.
    container.at(0)->setLabel("D1 label");
    ASSERT_EQ(spy.count, 1);
    container.at(1)->setLabel("D2 label");
    ASSERT_EQ(spy.count, 2);

    // Check that label content is correct
    ASSERT_EQ( QVariant{"D1 label"}, model.data(model.index(0,0), qcm::ContainerModel::ItemLabelRole) );
    ASSERT_EQ( QVariant{"D2 label"}, model.data(model.index(1,0), qcm::ContainerModel::ItemLabelRole) );

    // Dynamically modify label role
    spy.count = 0;
    model.setItemDisplayRole("dummyString");
    //ASSERT_EQ(2, spy.count);

    ASSERT_EQ( QVariant{"Dummy1"}, model.data(model.index(0,0), qcm::ContainerModel::ItemLabelRole) );
    ASSERT_EQ( QVariant{"Dummy2"}, model.data(model.index(1,0), qcm::ContainerModel::ItemLabelRole) );
}

TEST(qpsContainerModel, qObjectPtrItemDisplayRole)
{
    using Dummies = qcm::Container<QVector, QDummy*>;
    Dummies dummies;
    auto& model = *dummies.model();   // Force model creation...

    auto d1 = new QDummy{42., "Dummy1"};
    auto d2 = new QDummy{43., "Dummy2"};
    dummies.append(d1);
    dummies.append(d2);

    testItemDisplayRole(dummies);
}

TEST(qpsContainerModel, qObjectSharedPtrItemDisplayRole)
{
    using Dummies = qcm::Container<QVector, std::shared_ptr<QDummy>>;
    Dummies dummies;
    auto& model = *dummies.model();   // Force model creation...

    auto d1 = std::make_shared<QDummy>(42., "Dummy1");
    auto d2 = std::make_shared<QDummy>(43., "Dummy2");
    dummies.append(d1);
    dummies.append(d2);

    testItemDisplayRole(dummies);
}


template <class Container>
void    testWeakItemDisplayRole(Container& container)
{
    auto& model = *container.model();
    DataChangedSignalSpy spy(model);
    ASSERT_EQ(0, spy.count);

    // Content is not monitored for changes _before_ beeing displayed,
    // simulate a view access to QAbstractItemModel::data()
    model.data(model.index(0,0), qcm::ContainerModel::ItemLabelRole);
    model.data(model.index(1,0), qcm::ContainerModel::ItemLabelRole);
    ASSERT_EQ(0, spy.count);

    // Now modify container model qobject properties used as display role.
    container.at(0).lock()->setLabel("D1 label");
    ASSERT_EQ(spy.count, 1);
    container.at(1).lock()->setLabel("D2 label");
    ASSERT_EQ(spy.count, 2);

    // Check that label content is correct
    ASSERT_EQ( QVariant{"D1 label"}, model.data(model.index(0,0), qcm::ContainerModel::ItemLabelRole) );
    ASSERT_EQ( QVariant{"D2 label"}, model.data(model.index(1,0), qcm::ContainerModel::ItemLabelRole) );

    // Dynamically modify label role
    spy.count = 0;
    model.setItemDisplayRole("dummyString");
    //ASSERT_EQ(2, spy.count);

    ASSERT_EQ( QVariant{"Dummy1"}, model.data(model.index(0,0), qcm::ContainerModel::ItemLabelRole) );
    ASSERT_EQ( QVariant{"Dummy2"}, model.data(model.index(1,0), qcm::ContainerModel::ItemLabelRole) );
}

TEST(qpsContainerModel, qObjectWeakPtrItemDisplayRole)
{
    using   WeakDummy = std::weak_ptr<QDummy>;
    using Dummies = qcm::Container<QVector, std::weak_ptr<QDummy>>;
    Dummies dummies;
    auto& model = *dummies.model();   // Force model creation...

    auto sd1 = std::make_shared<QDummy>(42., "Dummy1");
    auto sd2 = std::make_shared<QDummy>(43., "Dummy2");

    auto d1 = WeakDummy{sd1};
    auto d2 = WeakDummy{sd2};
    dummies.append(d1);
    dummies.append(d2);

    testWeakItemDisplayRole(dummies);
}


// Test QObject destroyed signal monitoring (a QObject deleted from qt side outside
// QCM interface should be automatically removed from a QCM container
TEST(qpsContainer, qVectorQObjectPtrDestroyed)
{
    using QObjects = qcm::Container<QVector, QObject*>;
    QObjects objects;
    const auto o1 = new QObject{nullptr};
    objects.append( o1 );
    objects.append( new QObject{nullptr} );
    EXPECT_EQ( objects.size(), 2 );
    delete o1;                          // Expecting o1 to be automatically removed from container even when deleted outside of QCM interface
    EXPECT_EQ( objects.size(), 1 );

    // Note: Testing against std::shared_ptr<QObject> or std::weak_ptr<QObject> has no sense: destruction of such a managed
    // pointer throw a system exception... (linux g++7...)
}

/*
// std::weak_ptr<QObject> container model
TEST(qpsContainerModel, weakQObjectListReferenceItemAtEmpty)
{
    // Expect invalid return for invalid input arguments
    using WeakQObjects = qcm::ContainerModel< QVector, std::weak_ptr<QObject> >;
    WeakQObjects objects;
    auto listRef = objects.getListReference();
    ASSERT_TRUE( listRef != nullptr );
    ASSERT_TRUE( ( listRef->itemAt(-1) == QVariant{} ) );   // EXPECT -1, argument is invalid
    ASSERT_TRUE( ( listRef->itemAt(0) == QVariant{} ) );    // EXPECT invalid return QVariant, index is invalid

    ASSERT_TRUE( ( listRef->at(-1) == nullptr ) );   // EXPECT nullptr, argument is invalid
    ASSERT_TRUE( ( listRef->at(0) == nullptr ) );    // EXPECT nullptr, index is invalid
}

using WeakQObject = std::weak_ptr<QObject>;
Q_DECLARE_METATYPE( WeakQObject );

TEST(qpsContainerModel, weakQObjectPtrListReferenceItemAt)
{
    using WeakQObject = std::weak_ptr<QObject>;
    using WeakQObjects = qcm::ContainerModel< QVector, std::weak_ptr<QObject> >;
    WeakQObjects objects;
    auto o1Ptr{std::make_shared<QObject>()};
    auto o1{WeakQObject{o1Ptr}};
    auto o2Ptr{std::make_shared<QObject>()};
    auto o2{WeakQObject{o2Ptr}};
    objects.append( o1 );
    auto o3Ptr{std::make_shared<QObject>()};
    objects.append( WeakQObject{o3Ptr} );
    objects.append( o2 );

    auto listRef = objects.getListReference();
    ASSERT_TRUE( listRef != nullptr );
    ASSERT_TRUE( ( listRef->itemAt(0) != QVariant{} ) );
    EXPECT_TRUE( ( qvariant_cast<QObject*>( listRef->itemAt(0) ) == o1.lock().get() ) );
    EXPECT_TRUE( ( qvariant_cast<QObject*>( listRef->itemAt(2) ) == o2.lock().get() ) );
    ASSERT_TRUE( ( listRef->itemAt(3) == QVariant{} ) );

    ASSERT_TRUE( ( listRef->at(0) != nullptr ) );
    EXPECT_TRUE( ( listRef->at(0) == o1.lock().get() ) );
    EXPECT_TRUE( ( listRef->at(2) == o2.lock().get() ) );
    ASSERT_TRUE( ( listRef->at(3) == nullptr ) );
}
*/
