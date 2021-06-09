#include <bits/stdc++.h>

using std::cout;
using std::endl;

using Id = std::string;

struct IObject {
    virtual ~IObject() = default;
};

std::map<Id, std::unique_ptr<IObject>> objects;

struct INode {
    Id myname;
    std::map<Id, Id> inputs;
    std::map<Id, Id> outputs;

    virtual void apply() = 0;

    IObject *get_input(Id const &id) {
        return objects.at(inputs.at(id)).get();
    }

    template <class T>
        T *get_input(Id const &id) {
            return dynamic_cast<T *>(get_input(id));
        }

    void set_output(Id const &id, std::unique_ptr<IObject> &&obj) {
        auto objid = myname + "::" + id;
        objects[objid] = std::move(obj);
        outputs[id] = objid;
    }
};

std::map<Id, std::unique_ptr<INode>> nodes;

struct ParamDescriptor {
  std::string type, name, defl;

  ParamDescriptor(std::string const &type,
	  std::string const &name, std::string const &defl)
      : type(type), name(name), defl(defl) {}
};

template <class S, class T>
static std::string join_str(std::vector<T> const &elms, S const &delim) {
  std::stringstream ss;
  auto p = elms.begin(), end = elms.end();
  if (p != end)
    ss << *p++;
  for (; p != end; ++p) {
    ss << delim << *p;
  }
  return ss.str();
}

struct Descriptor {
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::vector<ParamDescriptor> params;
  std::vector<std::string> categories;

  Descriptor() = default;
  Descriptor(
	  std::vector<std::string> inputs,
	  std::vector<std::string> outputs,
	  std::vector<ParamDescriptor> params,
	  std::vector<std::string> categories)
      : inputs(inputs), outputs(outputs), params(params), categories(categories) {}

  std::string serialize() const {
      std::string res = "";
      res += "(" + join_str(inputs, ",") + ")";
      res += "(" + join_str(outputs, ",") + ")";
      std::vector<std::string> paramStrs;
      for (auto const &[type, name, defl] : params) {
          paramStrs.push_back(type + ":" + name + ":" + defl);
      }
      res += "(" + join_str(paramStrs, ",") + ")";
      res += "(" + join_str(categories, ",") + ")";
      return res;
  }
};

struct INodeClass {
    std::unique_ptr<Descriptor> desc;

    INodeClass(Descriptor const &desc)
        : desc(std::make_unique<Descriptor>(desc)) {}

    virtual std::unique_ptr<INode> new_instance() const = 0;
};

template <class F>
struct ImplNodeClass : INodeClass {
    F const &ctor;

    ImplNodeClass(F const &ctor, Descriptor const &desc)
        : INodeClass(desc), ctor(ctor) {}

    virtual std::unique_ptr<INode> new_instance() const override {
        return ctor();
    }
};

std::map<Id, std::unique_ptr<INodeClass>> nodeClasses;

template <class F>
int defNodeClass(F const &ctor, Id const &id, Descriptor const &desc = {}) {
    nodeClasses[id] = std::make_unique<ImplNodeClass<F>>(ctor, desc);
    return 1;
}

void addNode(Id const &cls, Id const &id) {
    auto node = nodeClasses.at(cls)->new_instance();
    node->myname = id;
    nodes[id] = std::move(node);
}

void applyNode(Id const &id) {
    nodes.at(id)->apply();
}

void setNodeInput(Id const &dn, Id const &ds, Id const &sn, Id const &ss) {
    nodes.at(dn)->inputs[ds] = nodes.at(sn)->outputs.at(ss);
}

std::string dumpDescriptors() {
  std::string res = "";
  for (auto const &[key, cls] : nodeClasses) {
    res += key + ":" + cls->desc->serialize() + "\n";
  }
  return res;
}

struct MyObject : IObject {
    int i = 0;
};

struct MyNodeA : INode {
    virtual void apply() override {
        auto obj = std::make_unique<MyObject>();
        set_output("Out0", std::move(obj));
    }
};

struct MyNodeB : INode {
    virtual void apply() override {
        auto obj = get_input<MyObject>("In0");
        auto newobj = std::make_unique<MyObject>();
        newobj->i = obj->i + 1;
        set_output("Out0", std::move(newobj));
    }
};

int defMyNodeA = defNodeClass(std::make_unique<MyNodeA>, "MyNodeA");
int defMyNodeB = defNodeClass(std::make_unique<MyNodeB>, "MyNodeB");

int main()
{
    addNode("MyNodeA", "A");
    addNode("MyNodeB", "B");
    applyNode("A");
    setNodeInput("B", "In0", "A", "Out0");
    applyNode("B");
    auto objid = nodes.at("B")->outputs.at("Out0");
    cout << "objid=" << objid << endl;
    auto obj = dynamic_cast<MyObject *>(objects.at(objid).get());
    cout << "obj->i=" << obj->i << endl;
    return 0;
}
