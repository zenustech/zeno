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
// \file	qanDataFlow.h
// \author	benoit@destrat.io
// \date	2016 12 12
//-----------------------------------------------------------------------------

#ifndef qanDataFlow_h
#define qanDataFlow_h

// QuickQanava headers
#include <QuickQanava>

// Qt headers
#include <QQuickPaintedItem>

namespace qan { // ::qan

class FlowNodeBehaviour : public qan::NodeBehaviour
{
    Q_OBJECT
public:
    explicit FlowNodeBehaviour(QObject* parent = nullptr) : qan::NodeBehaviour{ "FlowNodeBehaviour", parent } { /* Nil */ }
protected:
    virtual void    inNodeInserted( qan::Node& inNode, qan::Edge& edge ) noexcept override;
    virtual void    inNodeRemoved( qan::Node& inNode, qan::Edge& edge ) noexcept override;
};

class FlowNode : public qan::Node
{
    Q_OBJECT
public:
    enum class Type {
        Percentage,
        Image,
        Operation,
        Color,
        Tint
    };
    Q_ENUM(Type)

    explicit FlowNode( QQuickItem* parent = nullptr ) : FlowNode( Type::Percentage, parent ) {}
    explicit FlowNode( Type type, QQuickItem* parent = nullptr ) :
        qan::Node{parent}, _type{type} { /* Nil */ }
    virtual ~FlowNode() { /* Nil */ }

    FlowNode(const FlowNode&) = delete;
    FlowNode& operator=(const FlowNode&) = delete;
    FlowNode(FlowNode&&) = delete;
    FlowNode& operator=(FlowNode&&) = delete;

    static  QQmlComponent*      delegate(QQmlEngine& engine) noexcept;

public:
    Q_PROPERTY(Type type READ getType CONSTANT FINAL)
    inline  Type    getType() const noexcept { return _type; }
protected:
    Type            _type{Type::Percentage};

public slots:
    virtual void    inNodeOutputChanged();

public:
    Q_PROPERTY(QVariant output READ getOutput WRITE setOutput NOTIFY outputChanged)
    inline QVariant getOutput() const noexcept { return _output; }
    void            setOutput(QVariant output) noexcept;
protected:
    QVariant        _output;
signals:
    void            outputChanged();
};

class PercentageNode : public qan::FlowNode
{
    Q_OBJECT
public:
    PercentageNode() : qan::FlowNode{FlowNode::Type::Percentage} { setOutput(0.); }
    static  QQmlComponent*      delegate(QQmlEngine& engine) noexcept;
};

class OperationNode : public qan::FlowNode
{
    Q_OBJECT
public:
    enum class Operation {
        Add,
        Multiply
    };
    Q_ENUM(Operation)

    OperationNode() : qan::FlowNode{FlowNode::Type::Operation} {
        // When user change operation, recompute an output value
        connect(this, &OperationNode::operationChanged, this, &FlowNode::inNodeOutputChanged);
    }
    static  QQmlComponent*      delegate(QQmlEngine& engine) noexcept;

    Q_PROPERTY(Operation operation READ getOperation WRITE setOperation NOTIFY operationChanged)
    inline Operation    getOperation() const noexcept { return _operation; }
    void                setOperation(Operation operation) noexcept;
private:
    Operation           _operation{Operation::Multiply};
signals:
    void                operationChanged();

protected slots:
    void                inNodeOutputChanged();
};

class ImageNode : public qan::FlowNode
{
    Q_OBJECT
public:
    ImageNode() : qan::FlowNode{FlowNode::Type::Image} { setOutput(QStringLiteral("qrc:/Lenna.jpeg")); }
    static  QQmlComponent*      delegate(QQmlEngine& engine) noexcept;
};

class ColorNode : public qan::FlowNode
{
    Q_OBJECT
public:
    ColorNode() : qan::FlowNode{FlowNode::Type::Color} { setOutput(QColor{Qt::darkBlue}); }
    static  QQmlComponent*      delegate(QQmlEngine& engine) noexcept;
};

class TintNode : public qan::FlowNode
{
    Q_OBJECT
public:
    TintNode() : qan::FlowNode{FlowNode::Type::Tint} { }
    static  QQmlComponent*      delegate(QQmlEngine& engine) noexcept;

    Q_PROPERTY(QUrl source READ getSource WRITE setSource NOTIFY sourceChanged)
    inline QUrl     getSource() const noexcept { return _source; }
    void            setSource(QUrl source) noexcept;
private:
    QUrl            _source;
signals:
    void            sourceChanged();
public:
    Q_PROPERTY(QColor tintColor READ getTintColor WRITE setTintColor NOTIFY tintColorChanged)
    inline QColor   getTintColor() const noexcept { return _tintColor; }
    void            setTintColor(QColor tintColor) noexcept;
private:
    QColor          _tintColor{Qt::transparent};
signals:
    void            tintColorChanged();

protected slots:
    void            inNodeOutputChanged();
};

class FlowGraph : public qan::Graph
{
    Q_OBJECT
public:
    explicit FlowGraph( QQuickItem* parent = nullptr ) noexcept : qan::Graph(parent) { }
public:
    Q_INVOKABLE qan::Node*  insertFlowNode(int type) { return insertFlowNode(static_cast<FlowNode::Type>(type)); }       // FlowNode::Type could not be used from QML, Qt 5.10 bug???
    qan::Node*              insertFlowNode(FlowNode::Type type);
};

} // ::qan

QML_DECLARE_TYPE( qan::FlowNode )
QML_DECLARE_TYPE( qan::FlowGraph )
Q_DECLARE_METATYPE( qan::FlowNode::Type )
Q_DECLARE_METATYPE( qan::OperationNode::Operation )

#endif // qanDataFlow_h

