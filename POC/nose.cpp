#include <bits/stdc++.h>

using std::cout;
using std::endl;

using Id = std::string;

struct IObject {
  virtual ~IObject() = default;
};

std::map<Id, std::unique_ptr<IObject>> objects;

struct INode {
  std::map<Id, Id> inputs;
  std::map<Id, Id> outputs;

  virtual void apply() = 0;
};

std::map<Id, std::unique_ptr<INode>> nodes;

void addNode(Id const &id, std::unique_ptr<INode> &&node) {
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
    objects["A::TestOut"] = std::move(obj);
    outputs["Out0"] = "A::TestOut";
  }
};

struct MyNodeB : INode {
  virtual void apply() override {
    auto obj = dynamic_cast<MyObject *>(objects.at(inputs.at("In0")).get());
    auto newobj = std::make_unique<MyObject>();
    newobj->i = obj->i + 1;
    objects["B::TestOut"] = std::move(newobj);
    outputs["Out0"] = "B::TestOut";
  }
};

int main()
{
  addNode("A", std::make_unique<MyNodeA>());
  addNode("B", std::make_unique<MyNodeB>());
  applyNode("A");
  setNodeInput("B", "In0", "A", "Out0");
  applyNode("B");
  auto objid = nodes.at("B")->outputs["Out0"];
  cout << "objid=" << objid << endl;
  auto obj = dynamic_cast<MyObject *>(objects.at(objid).get());
  cout << "obj->i=" << obj->i << endl;
  return 0;
}
