#ifndef __API_INTERFACE_H__
#define __API_INTERFACE_H__

#include <memory>
#include <string>
#include <variant>

typedef std::variant<std::string, int, float, double, bool> ZVARIANT;

class IZNode
{
public:
    virtual std::string getName() const = 0;
    virtual std::string getIdent() const = 0;
    virtual ZVARIANT getSocketDefl(const std::string& sockName) = 0;
    virtual void setSocketDefl(const std::string& sockName, const ZVARIANT& value) = 0;
    virtual ZVARIANT getParam(const std::string& name) = 0;
    virtual void setParamValue(const std::string& name, const ZVARIANT& value) = 0;
};

class IZSubgraph
{
public:
    virtual ~IZSubgraph() = 0;
    virtual std::string name() const = 0;
    virtual std::shared_ptr<IZNode> getNode(const std::string& ident) = 0;
    virtual std::shared_ptr<IZNode> addNode(const std::string& nodeCls) = 0;

    virtual bool addLink(
        const std::string& outIdent,
        const std::string& outSock,
        const std::string& inIdent,
        const std::string& inSock) = 0;

    virtual bool removeLink(
        const std::string& outIdent,
        const std::string& outSock,
        const std::string& inIdent,
        const std::string& inSock) = 0;

    virtual int count() const = 0;
    virtual std::shared_ptr<IZNode> item(int idx) = 0;
};

class IZApplication
{
public:
    virtual void clear() = 0;
    virtual void openFile(const std::string& filePath) = 0;

    virtual int count() const = 0;
    virtual std::shared_ptr<IZSubgraph> item(int idx) = 0;

    virtual std::shared_ptr<IZSubgraph> getSubgraph(const std::string& name) = 0;
    virtual std::shared_ptr<IZSubgraph> addSubgraph(const std::string& name) = 0;
    virtual bool removeSubgraph(const std::string& name) = 0;
    virtual std::string forkSubgraph(const std::string& name) = 0;
};


#endif