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
// \file	custom.h
// \author	benoit@destrat.io
// \date	2017 03 19
//-----------------------------------------------------------------------------

#pragma once

// QuickQanava headers
#include "QuickQanava"

class CustomRectNode : public qan::Node
{
    Q_OBJECT
public:
    explicit CustomRectNode(QObject* parent=nullptr)  : qan::Node{parent} { /* Nil */ }
    virtual ~CustomRectNode() override = default;
    CustomRectNode(const CustomRectNode&) = delete;

public:
    static  QQmlComponent*  delegate(QQmlEngine& engine) noexcept;
    static  qan::NodeStyle* style(QObject* parent = nullptr) noexcept;
};

class CustomRoundNode : public qan::Node
{
    Q_OBJECT
public:
    explicit CustomRoundNode(QObject* parent=nullptr) : qan::Node{parent} { }
    virtual ~CustomRoundNode() override = default;
    CustomRoundNode( const CustomRoundNode& ) = delete;

public:
    static  QQmlComponent*  delegate(QQmlEngine& engine) noexcept;
    static  qan::NodeStyle* style(QObject* parent = nullptr) noexcept;
};

QML_DECLARE_TYPE(CustomRectNode)
QML_DECLARE_TYPE(CustomRoundNode)

class CustomEdge : public qan::Edge
{
    Q_OBJECT
public:
    explicit CustomEdge(QObject* parent = nullptr) : qan::Edge{parent} { }
    virtual ~CustomEdge() override = default;
    CustomEdge(const CustomEdge&) = delete;

public:
    static  QQmlComponent*  delegate(QQmlEngine& engine, QObject* parent = nullptr) noexcept;
    static  qan::EdgeStyle* style(QObject* parent = nullptr) noexcept;
};

QML_DECLARE_TYPE(CustomEdge)

class CustomGraph : public qan::Graph
{
    Q_OBJECT
public:
    explicit CustomGraph(QQuickItem* parent = nullptr) : qan::Graph(parent) { /* Nil */ }
    virtual ~CustomGraph() override = default;
    CustomGraph(const CustomGraph&) = delete;

public:
    Q_INVOKABLE qan::Node*  insertRectNode();
    Q_INVOKABLE qan::Node*  insertRoundNode();
    Q_INVOKABLE qan::Edge*  insertCustomEdge(qan::Node* source, qan::Node* destination);
};

QML_DECLARE_TYPE(CustomGraph)
