#pragma once

/*************************************************************************\
  Node editor - yet another artwork presented by archibate

  A node has these properties:

  1. node title, as displayed in the UI, possibly indicates its type
  2. input sockets, can be connected to an output socket of another node
  3. output sockets, can be connected to an input socket of another node
  4. parameters, these are immediate values, including:
  - float
  - int
  - string
  - float3
  - int3

  A link (edge) has these properties:

  1. source socket, an output socket of some node
  2. destination socket, an input socket of some node

  An input / output socket has these properties:

  1. socket name, indicate what does the socket means in a node
  2. socket type, indicate the data type flowing in this socket (WIP)

  Except from properties above, each of them also has a unique id to be
  distinguished from each other for both us and ImGui / imnodes.

  The node editor is currently header-only and may be used by any source
  not only limited to the OpenGL backend of ImGui which we currently use.

  Dependencies are: imgui, imnodes, glfw, glm

  To render the node editor using ImGui:

    auto editor = std::make_unique<NodeEditor>();
    ...
    while (mainloop) {
      ...
      ImGui::Begin("Editor Title");
      editor->draw();
      ImGui::End();
      ...
    }

  To load serialized node descriptors generated from NodeSystem:

  editor->load_descriptors(
        "MakeMatrix:(position,rotation,scale)(matrix)()\n"
        "ReadObjMesh:()(mesh)(string:path)\n"
        "TransformMesh:(mesh,matrix)(mesh)()\n"
        );

  To dump the UI nodes to file (as a Python script) for simulation apps:

    if (ImGui::Button("Save Button")) {
      std::fstream fout("simulation.py");
      editor->dump_graph(fout);
    }

  The dumped .py file will looks like this:

    import zen
    zen.addNode('ReadParticles', 'No1')
    zen.setNodeParam('No1', 'path', str('assets/monkey.obj'))
    zen.applyNode('No1')
    zen.addNode('DemoSolver', 'No4')
    zen.setNodeInput('No4', 'ini_pars', 'No1::output')
    zen.setNodeParam('No4', 'dt', float(0.1))
    zen.setNodeParam('No4', 'G', zen.float3(0, 0, 9))
    zen.applyNode('No4')
    zen.addNode('WriteParticles', 'No11')
    zen.setNodeInput('No11', 'pars', 'No4::pars')
    zen.setNodeParam('No11', 'path', str('/tmp/a.obj'))
    zen.applyNode('No11')

\*************************************************************************/

#include <Hg/OpenGL/stdafx.hpp>
#include <Hg/StrUtils.h>
#include <imgui/imgui.h>
#include <imnodes/imnodes.h>
#include <sstream>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <map>
#include <set>

namespace zeneditor {

struct NodeEditor {
  struct Depsgraph {
    std::vector<std::string> wanted;
    std::map<std::string, std::vector<std::string>> deps;
    std::set<std::string> visited;
    std::vector<std::string> applies;

    void touch(std::string const &name) {
      if (visited.find(name) != visited.end())
        return;
      visited.insert(name);
      for (auto const &depname: deps.at(name)) {
        touch(depname);
      }
      applies.push_back(name);
    }

    void compute() {
      for (auto const &name: wanted) {
        touch(name);
      }
    }
  };

  static int alloc_id() {
    static int top_id = 1;
    return top_id++;
  }

  struct Output {
    std::string name;
    int id;

    explicit Output(std::string name) : id(alloc_id()), name(name) {
    }

    void draw() {
      imnodes::BeginOutputAttribute(id);
      ImGui::Text(name.c_str());
      imnodes::EndOutputAttribute();
    }
  };

  struct Input {
    std::string name;
    int id;

    explicit Input(std::string name) : id(alloc_id()), name(name) {
    }

    void draw() {
      imnodes::BeginInputAttribute(id);
      ImGui::TextUnformatted(name.c_str());
      imnodes::EndInputAttribute();
    }
  };

  struct Value {
    std::string name;
    int id;

    explicit Value(std::string name = "") : id(alloc_id()), name(name) {
    }

    void draw() {
      imnodes::BeginStaticAttribute(id);
      ImGui::TextUnformatted(name.c_str());
      ImGui::SameLine();
      ImGui::SetNextItemWidth(ImGui::GetFontSize() * eval_width());
      draw_slider();
      imnodes::EndStaticAttribute();
    }

    virtual void draw_slider() = 0;
    virtual void dump(std::ostream &out) = 0;
    virtual void parse(std::istream &in) = 0;
    virtual std::string unparse(std::ostream &out) = 0;

    virtual int eval_width() {
      return 6;
    }

    template <class... Args>
    static std::unique_ptr<Value> make(std::string type,
        std::string name, std::string defl) {
      std::unique_ptr<Value> ret;
      if (type == "float") {
        ret = std::make_unique<FloatValue>();
      } else if (type == "int") {
        ret = std::make_unique<IntValue>();
      } else if (type == "string") {
        ret = std::make_unique<StringValue>();
      } else if (type == "float3") {
        ret = std::make_unique<Float3Value>();
      } else if (type == "int3") {
        ret = std::make_unique<Int3Value>();
      } else {
        printf("bad type name: %s\n", type.c_str());
        assert(0 && "bad type name");
      }
      ret->name = name;
      std::stringstream ss(defl);
      ret->parse(ss);
      return ret;
    }
  };

  struct FloatValue : Value {
    float value{0};

    float minval{0};
    float maxval{0};
    bool has_minval{false};
    bool has_maxval{false};

    virtual void draw_slider() override {
      ImGui::InputFloat("", &value);
      if (has_minval)
        value = std::max(value, minval);
      if (has_maxval)
        value = std::min(value, maxval);
    }

    virtual void dump(std::ostream &out) override {
      out << "float(" << value << ")";
    }

    virtual void parse(std::istream &in) override {
      in >> value;
      has_minval = bool(in >> minval);
      has_maxval = bool(in >> maxval);
    }

    virtual std::string unparse(std::ostream &out) override {
      out << value;
      return "float";
    }
  };

  struct IntValue : Value {
    int value{0};

    int minval{0};
    int maxval{0};
    bool has_minval{false};
    bool has_maxval{false};

    virtual void draw_slider() override {
      ImGui::InputInt("", &value);
      if (has_minval)
        value = std::max(value, minval);
      if (has_maxval)
        value = std::min(value, maxval);
    }

    virtual void dump(std::ostream &out) override {
      out << "int(" << value << ")";
    }

    virtual void parse(std::istream &in) override {
      in >> value;
      has_minval = bool(in >> minval);
      has_maxval = bool(in >> maxval);
    }

    virtual std::string unparse(std::ostream &out) override {
      out << value;
      return "int";
    }
  };

  struct Float3Value : Value {
    float value[3]{0, 0, 0};

    virtual void draw_slider() override {
      ImGui::InputFloat3("", value);
    }

    virtual int eval_width() override {
      return 10;
    }

    virtual void dump(std::ostream &out) override {
      out << "zen.float3(" << value[0] << ", " << value[1] << ", " << value[2] << ")";
    }

    virtual void parse(std::istream &in) override {
      in >> value[0] >> value[1] >> value[2];
    }

    virtual std::string unparse(std::ostream &out) override {
      out << value[0] << ' ' << value[1] << ' ' << value[2];
      return "float3";
    }
  };

  struct Int3Value : Value {
    int value[3]{0, 0, 0};

    virtual void draw_slider() override {
      ImGui::InputInt3("", value);
    }

    virtual int eval_width() override {
      return 10;
    }

    virtual void dump(std::ostream &out) override {
      out << "zen.int3(" << value[0] << ", " << value[1] << ", " << value[2] << ")";
    }

    virtual void parse(std::istream &in) override {
      in >> value[0] >> value[1] >> value[2];
    }

    virtual std::string unparse(std::ostream &out) override {
      out << value[0] << ' ' << value[1] << ' ' << value[2];
      return "int3";
    }
  };

  struct StringValue : Value {
    std::string value{""};

    virtual void draw_slider() override {
      std::vector<char> buf(value.size() + 1024);
      strcpy(buf.data(), value.c_str());
      ImGui::InputText("", buf.data(), buf.size());
      value = buf.data();
    }

    virtual int eval_width() override {
      return 10;
    }

    virtual void dump(std::ostream &out) override {
      out << "str(f'" << value << "')";
    }

    virtual void parse(std::istream &in) override {
      in >> value;
    }

    virtual std::string unparse(std::ostream &out) override {
      out << value;
      return "str";
    }
  };

  struct Node {
    std::string type;
    std::string name;
    int id;

    explicit Node(std::string type) : id(alloc_id()), type(type) {
      char buf[256] = "";
      sprintf(buf, "No%d", id);
      name = buf;
    }

    std::vector<std::unique_ptr<Input>> inputs;
    std::vector<std::unique_ptr<Output>> outputs;
    std::vector<std::unique_ptr<Value>> params;

    void draw() {
      imnodes::BeginNode(id);

      imnodes::BeginNodeTitleBar();
      ImGui::TextUnformatted(type.c_str());
      imnodes::EndNodeTitleBar();

      imnodes::BeginStaticAttribute(id);
      ImGui::SetNextItemWidth(ImGui::GetFontSize() * 6);
      std::vector<char> buf(name.size() + 1024);
      strcpy(buf.data(), name.c_str());
      ImGui::InputText("", buf.data(), buf.size());
      name = buf.data();
      imnodes::EndStaticAttribute();

      for (auto const &param: params) {
        param->draw();
      }
      for (auto const &socket: inputs) {
        socket->draw();
      }
      for (auto const &socket: outputs) {
        socket->draw();
      }

      imnodes::EndNode();
    }

    void set_pos(ImVec2 const &pos) {
      imnodes::SetNodeScreenSpacePos(id, pos);
    }

    ImVec2 get_pos() {
      return imnodes::GetNodeScreenSpacePos(id);
    }
  };

  struct Link {
    int first, second;
    int id;

    Link(int first, int second) : id(alloc_id()), first(first), second(second) {
    }
  };

  using FNodesType = std::vector<std::tuple<
        std::string, std::string, int,
        std::vector<int>, std::vector<int>,
        std::vector<std::string>, int, int
      >>;
  using FLinksType = std::vector<std::tuple<int, int, int>>;
  using FGraphType = std::tuple<FNodesType, FLinksType>;

  void load_graph(FGraphType const &fGraph) {
    nodes.clear();
    links.clear();

    auto [fNodes, fLinks] = fGraph;

    for (auto const &[type, name, id, inputs, outputs,
        params, posx, posy]: fNodes) {
      std::unique_ptr<Node> node = nullptr;
      for (auto const &ty: types) {
        if (ty.name == type) {
          node = ty.make_node();
        }
      }
      assert(node);

      node->type = type;
      node->name = name;
      node->id = id;
      for (int i = 0; i < inputs.size(); i++) {
        node->inputs[i]->id = inputs[i];
      }
      for (int i = 0; i < outputs.size(); i++) {
        node->outputs[i]->id = outputs[i];
      }
      for (int i = 0; i < params.size(); i++) {
        std::stringstream ss(params[i]);
        node->params[i]->parse(ss);
      }

      node->set_pos(ImVec2(posx, posy));
      nodes[id] = std::move(node);
    }

    for (auto const &[id, first, second]: fLinks) {
      auto link = std::make_unique<Link>(first, second);
      link->id = id;
      links[id] = std::move(link);
    }
  }

  FGraphType save_graph() {
    FNodesType fNodes;
    FLinksType fLinks;

    for (auto const &[key, node]: nodes) {
      std::vector<int> inputs;
      std::vector<int> outputs;
      std::vector<std::string> params;
      for (auto const &i: node->params) {
        std::stringstream ss;
        i->unparse(ss);
        params.push_back(ss.str());
      }
      for (auto const &i: node->inputs) {
        inputs.push_back(i->id);
      }
      for (auto const &i: node->outputs) {
        outputs.push_back(i->id);
      }
      auto pos = node->get_pos();
      fNodes.push_back(
        std::make_tuple(node->type, node->name, node->id,
          inputs, outputs, params, pos.x, pos.y));
    }
    for (auto const &[key, link]: links) {
      fLinks.push_back(
        std::make_tuple(link->id, link->first, link->second));
    }

    return std::make_tuple(fNodes, fLinks);
  }

  std::map<int, std::unique_ptr<Node>> nodes;
  std::map<int, std::unique_ptr<Link>> links;

  void dump_graph(std::ostream &out) {
    std::map<int, int> dst2src;
    std::map<int, std::string> out2name;
    std::map<int, std::string> out2nodename;
    std::map<std::string, int> name2id;

    for (auto const &[key, link]: links) {
      dst2src[link->second] = link->first;
    }

    for (auto const &[key, node]: nodes) {
      name2id[node->name] = key;
      bool first_output = true;
      for (auto const &socket: node->outputs) {
        out2nodename[socket->id] = node->name;
        out2name[socket->id] = node->name + "::" + socket->name;
        first_output = false;
      }
    }

    Depsgraph dg;
    for (int key: get_selected_nodes()) {  // only selected nodes will be evaluated
      dg.wanted.push_back(nodes.at(key)->name);
    }

    // portal dependency mock
    std::map<std::string, std::string> portid2outnode;
    for (auto const &[key, node]: nodes) {
      if (node->type == "PortalIn") {
        auto idname = dynamic_cast<StringValue *>(node->params[0].get())->value;
        portid2outnode[idname] = node->name;
      }
    }
    for (auto const &[key, node]: nodes) {
      auto &deps = dg.deps[node->name];
      if (node->type == "PortalOut") {
        auto idname = dynamic_cast<StringValue *>(node->params[0].get())->value;
        auto it = portid2outnode.find(idname);
        if (it == portid2outnode.end()) {
          printf("WARNING: PortalIn not found for %s\n", idname.c_str());
          continue;
        }
        deps.push_back(it->second);
      }
    }

    for (auto const &[key, node]: nodes) {
      auto &deps = dg.deps[node->name];
      for (auto const &socket: node->inputs) {
        if (dst2src.find(socket->id) == dst2src.end())
          continue;
        deps.push_back(out2nodename.at(dst2src.at(socket->id)));
      }
    }
    dg.compute();

    out << "def execute(frame):" << std::endl;
    out << "\timport zen" << std::endl;
    out << "\tif frame == 0: zen.addNode('EndFrame', 'endFrame')" << std::endl;

    for (auto const &name: dg.applies) {
      auto node = nodes.at(name2id.at(name)).get();

      out << "\tif frame == 0: zen.addNode('"  // only addNode on first frame
        << node->type << "', '" << node->name << "')" << std::endl;

      for (auto const &socket: node->inputs) {
        if (dst2src.find(socket->id) == dst2src.end())
          continue;

        auto dst_name = node->name + "', '" + socket->name;
        auto src_name = out2name.at(dst2src.at(socket->id));

        out << "\tzen.setNodeInput('"
          << dst_name << "', '" << src_name << "')" << std::endl;
      }

      for (auto const &param: node->params) {
        out << "\tzen.setNodeParam('"
          << node->name << "', '" << param->name << "', ";
        param->dump(out);
        out << ")" << std::endl;
      }

      out << "\tzen.applyNode('" << name << "')" << std::endl;
    }

    out << "\tzen.applyNode('endFrame')" << std::endl;
  }

  std::vector<int> get_selected_links() {
    std::vector<int> ret;
    int num = imnodes::NumSelectedLinks();
    if (num > 0) {
      ret.resize(num);
      imnodes::GetSelectedLinks(ret.data());
    }
    return ret;
  }

  std::vector<int> get_selected_nodes() {
    std::vector<int> ret;
    int num = imnodes::NumSelectedNodes();
    if (num > 0) {
      ret.resize(num);
      imnodes::GetSelectedNodes(ret.data());
    }
    return ret;
  }

  void draw() {
    imnodes::BeginNodeEditor();

    for (auto const &[key, node]: nodes) {
      node->draw();
    }
    for (auto const &[key, link]: links) {
      imnodes::Link(link->id, link->first, link->second);
    }

    imnodes::EndNodeEditor();

    int start_id, end_id;
    if (imnodes::IsLinkCreated(&start_id, &end_id)) {
      // auto erase other links to this socket
      for (auto const &[key, link]: links) {
        if (link->second == end_id) {
          links.erase(key);
        }
      }
      auto link = std::make_unique<Link>(start_id, end_id);
      links[link->id] = std::move(link);
    }

    if (ImGui::IsKeyPressed(GLFW_KEY_DELETE)) {
      for (auto const &link_id: get_selected_links()) {
        links.erase(link_id);
      }
      for (auto const &node_id: get_selected_nodes()) {
        // auto erase links from / to this node
        auto const &node = nodes[node_id];
        for (auto const &socket: node->inputs) {
          for (auto const &[key, link]: links) {
            if (link->second == socket->id) {
              links.erase(key);
            }
          }
        }
        for (auto const &socket: node->outputs) {
          for (auto const &[key, link]: links) {
            if (link->first == socket->id) {
              links.erase(key);
            }
          }
        }
        nodes.erase(node_id);
      }
    }

    if (!ImGui::IsAnyItemHovered() && ImGui::IsKeyPressed(GLFW_KEY_TAB)) {
      ImGui::OpenPopup("select category");
    }

    select_category_popup();
  }

  struct ParamDescriptor {
    std::string type, name, defl;
  };

  // node type descriptor
  struct NodeTypeDesc {
    std::string name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<ParamDescriptor> params;
    std::set<std::string> categories;

    std::unique_ptr<Node> make_node() const {
      auto node = std::make_unique<Node>(name.c_str());
      for (auto const &[type, name, defl]: params) {
        node->params.push_back(Value::make(type, name, defl));
      }
      for (auto const &name: inputs) {
        node->inputs.push_back(std::make_unique<Input>(name));
      }
      for (auto const &name: outputs) {
        node->outputs.push_back(std::make_unique<Output>(name));
      }
      return node;
    }

    void deserialize(std::string const &line) {
      std::istringstream ss(line);
      std::string key, ins, outs, pars, cats;
      char chr;
      std::getline(ss, name, ':');

      chr = 0;
      ss >> chr;
      assert(chr == '(');
      std::getline(ss, ins, ')');
      inputs = hg::split_str(ins, ',');

      chr = 0;
      ss >> chr;
      assert(chr == '(');
      std::getline(ss, outs, ')');
      outputs = hg::split_str(outs, ',');

      chr = 0;
      ss >> chr;
      assert(chr == '(');
      std::getline(ss, pars, ')');
      params.clear();
      for (auto const &s: hg::split_str(pars, ',')) {
        auto v = hg::split_str(s, ':');
        while (v.size() < 3)  // FIXME: why "string:path:" cause split_str error
          v.push_back("");
        assert(v.size() == 3);
        params.push_back(ParamDescriptor{v[0], v[1], v[2]});
      }

      chr = 0;
      ss >> chr;
      assert(chr == '(');
      std::getline(ss, cats, ')');
      for (auto const &s: hg::split_str(cats, ',')) {
        categories.insert(s);
      }
      if (categories.size() == 0)
        categories.insert("uncategorized");
    }
  };

  std::vector<NodeTypeDesc> types;

  void load_descriptors(std::string const &res) {
    types.clear();
    for (auto const &line: hg::split_str(res, '\n')) {
      NodeTypeDesc desc;
      desc.deserialize(line);
      for (auto const &cate: desc.categories) {
        categories.insert(cate);
      }
      types.push_back(desc);
    }
  }

  void new_node_popup(std::string cate) {
    auto click_pos = ImGui::GetMousePosOnOpeningCurrentPopup();

    for (auto const &type: types) {
      if (type.categories.find(cate) == type.categories.end())
        continue;
      if (ImGui::MenuItem(type.name.c_str())) {
        auto node = type.make_node();
        node->set_pos(click_pos);
        nodes[node->id] = std::move(node);
      }
    }
  }

  std::set<std::string> categories;

  void select_category_popup() {
    std::string category = "";

    if (ImGui::BeginPopup("select category")) {
      for (auto const &cate: categories) {
        if (ImGui::MenuItem(cate.c_str())) {
          category = cate;
          break;
        }
      }
      ImGui::EndPopup();
    }

    if (category.size())
      ImGui::OpenPopup(category.c_str());

    for (auto const &cate: categories) {
      if (ImGui::BeginPopup(cate.c_str())) {
        new_node_popup(cate);
        ImGui::EndPopup();
      }
    }
  }
};

}
