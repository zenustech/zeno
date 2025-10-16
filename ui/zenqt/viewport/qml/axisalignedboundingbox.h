/****************************************************************************
**
** Copyright (C) 2017 Klarälvdalens Datakonsult AB, a KDAB Group company.
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

#ifndef AXIS_ALIGNED_BOUNDING_BOX
#define AXIS_ALIGNED_BOUNDING_BOX

#include <QVector>
#include <QVector3D>

class QDebug;

class AxisAlignedBoundingBox
{
public:
    AxisAlignedBoundingBox();

    AxisAlignedBoundingBox(const QVector<QVector3D>& points);

    void update( const QVector<QVector3D>& points );

    QVector3D center() const { return m_center; }
    QVector3D radii() const { return m_radii; }

    QVector3D minPoint() const { return m_center - m_radii; }
    QVector3D maxPoint() const { return m_center + m_radii; }

    float xExtent() const { return 2.0f * m_radii.x(); }
    float yExtent() const { return 2.0f * m_radii.y(); }
    float zExtent() const { return 2.0f * m_radii.z(); }

    float maxExtent() const { return qMax( xExtent(), qMax( yExtent(), zExtent() ) ); }
    float minExtent() const { return qMin( xExtent(), qMin( yExtent(), zExtent() ) ); }

private:
    QVector3D m_center;
    QVector3D m_radii;
};

QDebug & operator<<(QDebug & stream, const AxisAlignedBoundingBox & bbox);

#endif // AXIS_ALIGNED_BOUNDING_BOX
