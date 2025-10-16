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

#include "camera.h"

Camera::Camera(QObject *parent)
    : QObject(parent)
    , m_azimuth(0.0)
    , m_elevation(15.0)
    , m_distance(15.0)
{
}

float Camera::azimuth() const
{
    return m_azimuth;
}

float Camera::distance() const
{
    return m_distance;
}

float Camera::elevation() const
{
    return m_elevation;
}

void Camera::setAzimuth(float azimuth)
{
    if (m_azimuth == azimuth)
        return;

    m_azimuth = azimuth;
    emit azimuthChanged(azimuth);
}

void Camera::setDistance(float distance)
{
    if (m_distance == distance)
        return;

    m_distance = distance;
    emit distanceChanged(distance);
}

void Camera::setElevation(float elevation)
{
    if (m_elevation == elevation)
        return;

    m_elevation = elevation;
    emit elevationChanged(elevation);
}

