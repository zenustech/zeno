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
// \file	custom.cpp
// \author	benoit@destrat.io
// \date	2017 03 19
//-----------------------------------------------------------------------------

// Qt headers
#include <QQmlEngine>
#include <QQmlComponent>

// QuickQanava headers
#include "../../src/qanGraph.h"
#include "./custom.h"

QQmlComponent*  CustomRectNode::delegate(QQmlEngine& engine) noexcept
{
    static std::unique_ptr<QQmlComponent>   customRectNode_delegate;
    if ( !customRectNode_delegate )
        customRectNode_delegate = std::make_unique<QQmlComponent>(&engine, "qrc:/RectNode.qml");
    return customRectNode_delegate.get();
}

qan::NodeStyle* CustomRectNode::style(QObject* parent) noexcept
{
    Q_UNUSED(parent)
    static std::unique_ptr<qan::NodeStyle>  customRectNode_style;
    if ( !customRectNode_style ) {
        customRectNode_style = std::make_unique<qan::NodeStyle>();
        customRectNode_style->setBackColor(QColor("#b254fb"));
    }
    return customRectNode_style.get();
}


QQmlComponent*  CustomRoundNode::delegate(QQmlEngine& engine) noexcept
{
    static std::unique_ptr<QQmlComponent>   customRoundNode_delegate;
    if ( !customRoundNode_delegate )
        customRoundNode_delegate = std::make_unique<QQmlComponent>(&engine, "qrc:/RoundNode.qml");
    return customRoundNode_delegate.get();
}

qan::NodeStyle* CustomRoundNode::style(QObject* parent) noexcept
{
    Q_UNUSED(parent)
    static std::unique_ptr<qan::NodeStyle>  customRoundNode_style;
    if ( !customRoundNode_style ) {
        customRoundNode_style = std::make_unique<qan::NodeStyle>();
        customRoundNode_style->setBackColor(QColor("#0770ff"));
    }
    return customRoundNode_style.get();
}


QQmlComponent*  CustomEdge::delegate(QQmlEngine& engine, QObject* parent) noexcept
{
    static std::unique_ptr<QQmlComponent>   customEdge_delegate;
    if (!customEdge_delegate)
        customEdge_delegate = std::make_unique<QQmlComponent>(&engine, "qrc:/CustomEdge.qml",
                                                              QQmlComponent::PreferSynchronous, parent);
    return customEdge_delegate.get();
}

qan::EdgeStyle* CustomEdge::style(QObject* parent) noexcept
{
    Q_UNUSED(parent)
    static std::unique_ptr<qan::EdgeStyle>  customEdge_style;
    if ( !customEdge_style )
        customEdge_style = std::make_unique<qan::EdgeStyle>();
    return customEdge_style.get();
}

qan::Node*  CustomGraph::insertRectNode()
{
    return qan::Graph::insertNode<CustomRectNode>();
}

qan::Node*  CustomGraph::insertRoundNode()
{
    return qan::Graph::insertNode<CustomRoundNode>();
}

qan::Edge*  CustomGraph::insertCustomEdge(qan::Node* source, qan::Node* destination)
{
    const auto engine = qmlEngine(this);
    if ( source != nullptr && destination != nullptr && engine != nullptr )
        return qan::Graph::insertEdge<CustomEdge>(*source, destination, CustomEdge::delegate(*engine) );
    return nullptr;
}
