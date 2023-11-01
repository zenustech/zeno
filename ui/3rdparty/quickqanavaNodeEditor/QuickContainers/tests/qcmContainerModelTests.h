/*
    This file is part of QuickContainers library.

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
// This file is a part of QuickContainers library.
//
// \file	qpsContainerModelTests.h
// \author	benoit@qanava.org
// \date	2016 02 08
//-----------------------------------------------------------------------------

// Qt headers

// QuickContainers headers
#include <QuickContainers>

class QDummy : public QObject
{
    Q_OBJECT
public:
    QDummy( QObject* parent = nullptr ) :
        QObject{ parent }
    {
        setObjectName("::QDummy");
    }
    QDummy( qreal dummyReal, const QString& dummyString, QObject* parent = nullptr ) :
        QObject{ parent },
        _dummyReal{dummyReal},
        _dummyString{dummyString}
    {
        setObjectName("::QDummy");
    }
    virtual ~QDummy() { /* Nil */ }
private:
    Q_DISABLE_COPY(QDummy)

public:
    Q_PROPERTY( qreal dummyReal READ getDummyReal WRITE setDummyReal NOTIFY dummyRealChanged )
    qreal       getDummyReal( ) const { return _dummyReal; }
    void        setDummyReal( qreal dummyReal ) { _dummyReal = dummyReal; emit dummyRealChanged(); }
protected:
    qreal       _dummyReal{ 42. };
signals:
    void        dummyRealChanged( );

public:
    Q_PROPERTY( QString label READ getLabel WRITE setLabel NOTIFY labelChanged )
    const QString&  getLabel() const { return _label; }
    void            setLabel(const QString& label ){ _label = label; emit labelChanged(); }
protected:
    QString         _label{QStringLiteral("Label")};
signals:
    void            labelChanged( );

public:
    Q_PROPERTY( QString dummyString READ getDummyString WRITE setDummyString NOTIFY dummyStringChanged )
    const QString&  getDummyString() const { return _dummyString; }
    void            setDummyString(const QString& dummyString ){ _dummyString = dummyString; emit dummyStringChanged(); }
protected:
    QString         _dummyString{QStringLiteral("Dummy")};
signals:
    void            dummyStringChanged( );
};

class DataChangedSignalSpy : public QObject
{
    Q_OBJECT
public:
    DataChangedSignalSpy(const QAbstractItemModel& target) :
        QObject{nullptr}
    {
        connect( &target,   &QAbstractItemModel::dataChanged,
                 this,      &DataChangedSignalSpy::onDataChanged );
    }

    int     count = 0;
public slots:
    void    onDataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight, const QVector<int> &roles = QVector<int> ())
    {
        Q_UNUSED(topLeft); Q_UNUSED(bottomRight); Q_UNUSED(roles);
        count++;
    }
};



