struct MyNode : zeno::INode {
    virtual void apply() override {
        // lit
        auto value = get_input2<int>("value");
        // num
        auto value = get_input<zeno::NumericObject>("value")->get<int>();


        // lit
        set_output2("value", 42);
        // num
        auto value = std::make_shared<zeno::NumericObject>();
        value->value = 42;
        set_output("value", value);
    }
};
