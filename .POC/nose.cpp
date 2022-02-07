#include <bits/stdc++.h>

using std::cout;
using std::endl;

struct IObject {
    virtual ~IObject() = default;
};

struct INode;
std::map<std::string, std::unique_ptr<IObject>> objects;
std::map<std::string, std::unique_ptr<INode>> nodes;

struct Context {
    std::set<std::string> visited;
};

void applyNode(std::string const &id, Context *ctx);

struct INode {
public:
    std::string myname;
    std::map<std::string, std::pair<std::string, std::string>> inputBounds;
    std::map<std::string, std::string> inputs;
    std::map<std::string, std::string> outputs;

    void doApply(Context *ctx) {
        for (auto [ds, bound]: inputBounds) {
            auto [sn, ss] = bound;
            applyNode(sn, ctx);
            inputs[ds] = nodes.at(sn)->outputs.at(ss);
        }
        apply();
    }

protected:
    /*
     * @name apply()
     * @brief user should override this pure virtual function,
     * @brief it will be called when the node is executed
     */
    virtual void apply() = 0;

    /*
     * @name has_input(id)
     * @param[id] the input socket name
     * @return true if connected, false otherwise
     * @brief test if the input socket is connected
     */
    bool has_input(std::string const &id) const {
        return objects.find(inputs.at(id)) != objects.end();
    }

    /*
     * @name get_input(id)
     * @param[id] the input socket name
     * @return pointer to the object
     * @brief get the object passed into the input socket
     */
    IObject *get_input(std::string const &id) const {
        return objects.at(inputs.at(id)).get();
    }

    /*
     * @name get_input<T>(id)
     * @template[T] the object type you want to cast to
     * @param[id] the input socket name
     * @return pointer to the object, will be null if the input is not of that type
     * @brief get the object passed into the input socket,
     * @brief and cast it to the given type
     */
    template <class T>
        T *get_input(std::string const &id) const {
            return dynamic_cast<T *>(get_input(id));
        }

    /*
     * @name set_output(id, std::move(obj))
     * @param[id] the output socket name
     * @param[obj] the (unique) pointer to the object
     * @brief set an object to the output socket
     */
    void set_output(std::string const &id, std::unique_ptr<IObject> &&obj) {
        auto objid = myname + "::" + id;
        objects[objid] = std::move(obj);
        outputs[id] = objid;
    }
};

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
	  std::vector<std::string> const &inputs,
	  std::vector<std::string> const &outputs,
	  std::vector<ParamDescriptor> const &params,
	  std::vector<std::string> const &categories)
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

std::map<std::string, std::unique_ptr<INodeClass>> nodeClasses;

template <class F>
int defNodeClass(F const &ctor, std::string const &id, Descriptor const &desc = {}) {
    nodeClasses[id] = std::make_unique<ImplNodeClass<F>>(ctor, desc);
    return 1;
}

void addNode(std::string const &cls, std::string const &id) {
    auto node = nodeClasses.at(cls)->new_instance();
    node->myname = id;
    nodes[id] = std::move(node);
}

void applyNode(std::string const &id, Context *ctx) {
    if (ctx->visited.find(id) != ctx->visited.end()) {
        return;
    }
    ctx->visited.insert(id);
    nodes.at(id)->doApply(ctx);
}

void bindNodeInput(std::string const &dn, std::string const &ds,
    std::string const &sn, std::string const &ss) {
    nodes.at(dn)->inputBounds[ds] = std::pair(sn, ss);
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

    ~MyObject() {
        printf("~MyObject() called, i = %d\n", i);
    }
};

struct MyNodeA : INode {
    virtual void apply() override {
        printf("MyNodeA::apply()\n");
        auto obj = std::make_unique<MyObject>();
        set_output("Out0", std::move(obj));
    }
};

struct MyNodeB : INode {
    virtual void apply() override {
        printf("MyNodeB::apply()\n");
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
    addNode("MyNodeB", "C");
    bindNodeInput("B", "In0", "A", "Out0");
    bindNodeInput("C", "In0", "B", "Out0");
    Context ctx;
    applyNode("C", &ctx);
    auto objid = nodes.at("C")->outputs.at("Out0");
    cout << "objid=" << objid << endl;
    auto obj = dynamic_cast<MyObject *>(objects.at(objid).get());
    cout << "obj->i=" << obj->i << endl;
    return 0;
}
