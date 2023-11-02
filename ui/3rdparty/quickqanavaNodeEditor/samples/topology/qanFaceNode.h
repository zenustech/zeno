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
// \file	qanFaceNode.h
// \author	benoit@destrat.io
// \date	2016 08 12
//-----------------------------------------------------------------------------

#pragma once

// QuickQanava headers
#include <QuickQanava>

// Qt headers
#include <QQuickPaintedItem>

namespace qan { // ::qan

class FaceNode : public qan::Node
{
    Q_OBJECT
public:
    explicit FaceNode(QQuickItem* parent = nullptr);
    virtual ~FaceNode() override = default;
private:
    Q_DISABLE_COPY(FaceNode)
public:
    Q_PROPERTY(QUrl image READ getImage WRITE setImage NOTIFY imageChanged)
    const QUrl&     getImage() const noexcept { return _image; }
    void            setImage(QUrl image) noexcept;
private:
    QUrl            _image;
signals:
    void            imageChanged();

    /*! \name Node Static Factories *///---------------------------------------
    //@{
public:
    static  QQmlComponent*      delegate(QQmlEngine& engine) noexcept;
    //@}
    //-------------------------------------------------------------------------
};

class FaceGraph : public qan::Graph
{
    Q_OBJECT
public:
    explicit FaceGraph(QQuickItem* parent = nullptr) noexcept :
        qan::Graph(parent) { }

public:
    Q_INVOKABLE qan::Node* insertFaceNode() {
        auto node = insertNode<FaceNode>(nullptr);
        return node;
    }
};

} // ::qan

QML_DECLARE_TYPE(qan::FaceNode)
QML_DECLARE_TYPE(qan::FaceGraph)
