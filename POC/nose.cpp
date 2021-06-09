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

void addNode(Id const &id, std::unique_ptr<INode> &&node) {
  node->myname = id;
  nodes[id] = std::move(node);
}

void applyNode(Id const &id) {
  nodes.at(id)->apply();
}

void setNodeInput(Id const &dn, Id const &ds, Id const &sn, Id const &ss) {
  nodes.at(dn)->inputs[ds] = nodes.at(sn)->outputs.at(ss);
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

int main()
{
  addNode("A", std::make_unique<MyNodeA>());
  addNode("B", std::make_unique<MyNodeB>());
  applyNode("A");
  setNodeInput("B", "In0", "A", "Out0");
  applyNode("B");
  auto objid = nodes.at("B")->outputs.at("Out0");
  cout << "objid=" << objid << endl;
  auto obj = dynamic_cast<MyObject *>(objects.at(objid).get());
  cout << "obj->i=" << obj->i << endl;
  return 0;
}
