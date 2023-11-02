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
// \file	qanDataFlow.cpp
// \author	benoit@destrat.io
// \date	2017 12 12
//-----------------------------------------------------------------------------

// QuickQanava headers
#include "../../src/QuickQanava.h"
#include "./qanDataFlow.h"

namespace qan { // ::qan

void    FlowNodeBehaviour::inNodeInserted( qan::Node& inNode, qan::Edge& edge ) noexcept
{
    Q_UNUSED(edge);
    const auto inFlowNode = qobject_cast<qan::FlowNode*>(&inNode);
    const auto flowNodeHost = qobject_cast<qan::FlowNode*>(getHost());
    if ( inFlowNode != nullptr &&
         flowNodeHost != nullptr ) {
        //
        QObject::connect(inFlowNode,    &qan::FlowNode::outputChanged,
                         flowNodeHost,  &qan::FlowNode::inNodeOutputChanged);
    }
    flowNodeHost->inNodeOutputChanged();    // Force a call since with a new edge insertion, actual value might aready be initialized
}

void    FlowNodeBehaviour::inNodeRemoved( qan::Node& inNode, qan::Edge& edge ) noexcept
{
    Q_UNUSED(inNode); Q_UNUSED(edge);
}

QQmlComponent*  FlowNode::delegate(QQmlEngine& engine) noexcept
{
    static std::unique_ptr<QQmlComponent>   qan_FlowNode_delegate;
    if ( !qan_FlowNode_delegate )
        qan_FlowNode_delegate = std::make_unique<QQmlComponent>(&engine, "qrc:/FlowNode.qml");
    return qan_FlowNode_delegate.get();
}

void    FlowNode::inNodeOutputChanged()
{

}

void    FlowNode::setOutput(QVariant output) noexcept
{
    _output = output;
    emit outputChanged();
}

QQmlComponent*  PercentageNode::delegate(QQmlEngine& engine) noexcept
{
    static std::unique_ptr<QQmlComponent>   delegate;
    if ( !delegate )
        delegate = std::make_unique<QQmlComponent>(&engine, "qrc:/PercentageNode.qml");
    return delegate.get();
}

QQmlComponent*  OperationNode::delegate(QQmlEngine& engine) noexcept
{
    static std::unique_ptr<QQmlComponent>   delegate;
    if ( !delegate )
        delegate = std::make_unique<QQmlComponent>(&engine, "qrc:/OperationNode.qml");
    return delegate.get();
}

void    OperationNode::setOperation(Operation operation) noexcept
{
    if (_operation != operation) {
        _operation = operation;
        emit operationChanged();
    }
}

void    OperationNode::inNodeOutputChanged()
{
    FlowNode::inNodeOutputChanged();
    qreal o = 0.; // For the example sake we do not deal with overflow
    bool oIsInitialized{false};
    for (const auto inNode : get_in_nodes()) {
        const auto inFlowNode = qobject_cast<qan::FlowNode*>(inNode);
        if (inFlowNode == nullptr ||
            !inFlowNode->getOutput().isValid())
            continue;
        bool ok = false;
        const auto inOutput = inFlowNode->getOutput().toReal(&ok);
        if (ok) {
            switch (_operation) {
            case Operation::Add:    o += inOutput; break;
            case Operation::Multiply:
                if (!oIsInitialized) {
                    o = inOutput;
                    oIsInitialized = true;
                } else
                    o *= inOutput;
                break;
            }
        }
    }
    setOutput(o);
}

QQmlComponent*  ImageNode::delegate(QQmlEngine& engine) noexcept
{
    static std::unique_ptr<QQmlComponent>   delegate;
    if ( !delegate )
        delegate = std::make_unique<QQmlComponent>(&engine, "qrc:/ImageNode.qml");
    return delegate.get();
}

QQmlComponent*  ColorNode::delegate(QQmlEngine& engine) noexcept
{
    static std::unique_ptr<QQmlComponent>   delegate;
    if ( !delegate )
        delegate = std::make_unique<QQmlComponent>(&engine, "qrc:/ColorNode.qml");
    return delegate.get();
}

QQmlComponent*  TintNode::delegate(QQmlEngine& engine) noexcept
{
    static std::unique_ptr<QQmlComponent>   delegate;
    if ( !delegate )
        delegate = std::make_unique<QQmlComponent>(&engine, "qrc:/TintNode.qml");
    return delegate.get();
}

void    TintNode::setSource(QUrl source) noexcept
{
    if ( _source != source ) {
        _source = source;
        emit sourceChanged();
    }
}

void    TintNode::setTintColor(QColor tintColor) noexcept
{
    if ( _tintColor != tintColor ) {
        _tintColor = tintColor;
        emit tintColorChanged();
    }
}

void    TintNode::inNodeOutputChanged()
{
    FlowNode::inNodeOutputChanged();
    qDebug() << "TintNode::inNodeOutputValueChanged()";
    if (get_in_nodes().size() != 3)
        return;

    // FIXME: Do not find port item by index, but by id with qan::NodeItem::findPort()...
    const auto inFactorNode = qobject_cast<qan::FlowNode*>(get_in_nodes().at(0));
    const auto inColorNode = qobject_cast<qan::FlowNode*>(get_in_nodes().at(1));
    const auto inImageNode = qobject_cast<qan::FlowNode*>(get_in_nodes().at(2));
    qDebug() << "inFactorNode=" << inFactorNode << "\tinColorNode=" << inColorNode << "\tinImageNode=" << inImageNode;
    if (inFactorNode == nullptr ||
        inColorNode == nullptr ||
        inImageNode == nullptr)
        return;
    bool factorOk = false;
    const auto factor = inFactorNode->getOutput().toReal(&factorOk);
    auto       tint =   inColorNode->getOutput().value<QColor>();
    const auto source = inImageNode->getOutput().toUrl();
    qDebug() << "factor=" << factor;
    qDebug() << "tint=" << tint;
    qDebug() << "source=" << source.toString();
    if (factorOk &&
        !source.isEmpty() &&
        tint.isValid()) {
        tint.setAlpha(qBound(0., factor, 1.0) * 255);
        setSource(source);
        setTintColor(tint);
    }
}

qan::Node* FlowGraph::insertFlowNode(FlowNode::Type type)
{
    qan::Node* flowNode = nullptr;
    switch ( type ) {
    case qan::FlowNode::Type::Percentage:
        flowNode = insertNode<PercentageNode>(nullptr);
        insertPort(flowNode, qan::NodeItem::Dock::Right, qan::PortItem::Type::Out, "OUT", "OUT" );
        break;
    case qan::FlowNode::Type::Image:
        flowNode = insertNode<ImageNode>(nullptr);
        insertPort(flowNode, qan::NodeItem::Dock::Right, qan::PortItem::Type::Out, "OUT", "OUT" );
        break;
    case qan::FlowNode::Type::Operation: {
        flowNode = insertNode<OperationNode>(nullptr);
        // Insert out port first we need to modify it from OperationNode.qml delegate
        insertPort(flowNode, qan::NodeItem::Dock::Right, qan::PortItem::Type::Out, "OUT", "OUT" );

        // In ports should have Single multiplicity: only one value (ie one input edge) binded to a port
        const auto inp1 = insertPort(flowNode, qan::NodeItem::Dock::Left, qan::PortItem::Type::In, "IN", "IN1" );
        inp1->setMultiplicity(qan::PortItem::Multiplicity::Single);
        const auto inp2 = insertPort(flowNode, qan::NodeItem::Dock::Left, qan::PortItem::Type::In, "IN", "IN2" );
        inp2->setMultiplicity(qan::PortItem::Multiplicity::Single);
    }
        break;
    case qan::FlowNode::Type::Color:
        flowNode = insertNode<ColorNode>(nullptr);
        insertPort(flowNode, qan::NodeItem::Dock::Right, qan::PortItem::Type::Out, "OUT", "OUT" );
        break;
    case qan::FlowNode::Type::Tint:
        flowNode = insertNode<TintNode>(nullptr);
        insertPort(flowNode, qan::NodeItem::Dock::Left, qan::PortItem::Type::In, "FACTOR", "FACTOR" );
        insertPort(flowNode, qan::NodeItem::Dock::Left, qan::PortItem::Type::In, "COLOR", "COLOR" );
        insertPort(flowNode, qan::NodeItem::Dock::Left, qan::PortItem::Type::In, "IMAGE", "IMAGE" );
        break;
    default: return nullptr;
    }
    if ( flowNode )
        flowNode->installBehaviour(std::make_unique<FlowNodeBehaviour>());
    return flowNode;
}

} // ::qan
