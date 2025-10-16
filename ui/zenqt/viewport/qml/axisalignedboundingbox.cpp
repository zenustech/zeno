/****************************************************************************
**
** Copyright (C) 2015 Klarälvdalens Datakonsult AB, a KDAB Group company.
** Author: Giuseppe D'Angelo
** Contact: info@kdab.com
**
** This program is free software: you can redistribute it and/or modify
** it under the terms of the GNU Lesser General Public License as published by
** the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
**
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU Lesser General Public License for more details.
**
** You should have received a copy of the GNU Lesser General Public License
** along with this program.  If not, see <http://www.gnu.org/licenses/>.
**
****************************************************************************/

#include "axisalignedboundingbox.h"

#include <QDebug>

AxisAlignedBoundingBox::AxisAlignedBoundingBox()
    : m_center(),
      m_radii()
{
}

AxisAlignedBoundingBox::AxisAlignedBoundingBox( const QVector<QVector3D>& points )
{
    update( points );
}

void AxisAlignedBoundingBox::update( const QVector<QVector3D>& points )
{
    if (points.isEmpty()) {
        m_center = QVector3D();
        m_radii = QVector3D();
        return;
    }

    QVector3D minPoint = points.at( 0 );
    QVector3D maxPoint = points.at( 0 );

    for ( int i = 1; i < points.size(); ++i )
    {
        const QVector3D& point = points.at( i );
        if ( point.x() > maxPoint.x() )
            maxPoint.setX( point.x() );
        if ( point.y() > maxPoint.y() )
            maxPoint.setY( point.y() );
        if ( point.z() > maxPoint.z() )
            maxPoint.setZ( point.z() );
        if ( point.x() < minPoint.x() )
            minPoint.setX( point.x() );
        if ( point.y() < minPoint.y() )
            minPoint.setY( point.y() );
        if ( point.z() < minPoint.z() )
            minPoint.setZ( point.z() );
    }

    m_center = 0.5 * ( minPoint + maxPoint );
    m_radii = 0.5 * ( maxPoint - minPoint );
#if 0
    qDebug() << "AABB:";
    qDebug() << "    min =" << minPoint;
    qDebug() << "    max =" << maxPoint;
    qDebug() << " center =" << m_center;
    qDebug() << "  radii =" << m_radii;
#endif
}


QDebug &operator<<(QDebug &stream, const AxisAlignedBoundingBox &bbox)
{
    stream << "AABB: min=" << bbox.minPoint() << ", max=" << bbox.maxPoint();
    return stream;
}
