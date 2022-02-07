#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <stack>
#include <map>


enum class Opcode {
    LOAD_ATTR = 0,
    STORE_ATTR = 1,
    UNARY_OP = 2,
    BINARY_OP = 3,
};


enum class BinaryOp {
    ADD, SUB,
};


enum class UnaryOp {
    COPY, NEG,
};


using ValueType = float;


struct Primitive {
    std::vector<ValueType> mAttrValues;

    Primitive() {
    }

    ValueType const &at(int index) const {
        return mAttrValues[index];
    }

    ValueType &at(int index) {
        return mAttrValues[index];
    }
};


struct Statement {
    std::vector<std::string> mOpStrings;
    std::vector<std::pair<Opcode, int>> mOps;
    std::map<std::string, int> mAttrNameToIndex;

    int attrNameToIndex(std::string const &name) const {
        auto it = mAttrNameToIndex.find(name);
        if (it == mAttrNameToIndex.end()) {
            printf("ERROR: attribute `%s` not found\n", name.c_str());
            abort();
        }
        return it->second;
    }

    int unaryOpNameToIndex(std::string const &name) {
        if (0) {
        } else if (name == "copy") {
            return (int)UnaryOp::COPY;
        } else if (name == "neg") {
            return (int)UnaryOp::NEG;
        } else {
            printf("ERROR: invalid operator name `%s`\n", name.c_str());
            return -1;
        }
    }

    int binaryOpNameToIndex(std::string const &name) {
        if (0) {
        } else if (name == "add") {
            return (int)BinaryOp::ADD;
        } else if (name == "sub") {
            return (int)BinaryOp::SUB;
        } else {
            printf("ERROR: invalid operator name `%s`\n", name.c_str());
            return -1;
        }
    }

    static ValueType applyUnaryOp(UnaryOp op, ValueType const &lhs) {
        switch (op) {
        case UnaryOp::COPY:
            return lhs;
        case UnaryOp::NEG:
            return -lhs;
        }
        return ValueType(0);
    }

    static ValueType applyBinaryOp(BinaryOp op, ValueType const &lhs, ValueType const &rhs) {
        switch (op) {
        case BinaryOp::ADD:
            return lhs + rhs;
        case BinaryOp::SUB:
            return lhs - rhs;
        }
        return ValueType(0);
    }

    void parse(std::string const &str) {
        std::stringstream ss(str);
        std::string s;
        while (ss >> s) {
            if (0) {

            } else if (s[0] == '@') {
                int index = attrNameToIndex(s.substr(1));
                mOps.emplace_back(Opcode::LOAD_ATTR, index);

            } else if (s[0] == '=') {
                int index = attrNameToIndex(s.substr(1));
                mOps.emplace_back(Opcode::STORE_ATTR, index);

            } else if (s[0] == '+') {
                int index = binaryOpNameToIndex(s.substr(1));
                mOps.emplace_back(Opcode::BINARY_OP, index);

            } else if (s[0] == '-') {
                int index = unaryOpNameToIndex(s.substr(1));
                mOps.emplace_back(Opcode::UNARY_OP, index);

            }
        }
    }

    void execute(Primitive *prim) const {
        std::stack<ValueType> valStack;

        for (auto const &op: mOps) {
            if (0) {

            } else if (op.first == Opcode::LOAD_ATTR) {
                auto value = prim->at(op.second);
                valStack.push(value);

            } else if (op.first == Opcode::STORE_ATTR) {
                auto value = valStack.top(); valStack.pop();
                prim->at(op.second) = value;

            } else if (op.first == Opcode::UNARY_OP) {
                auto lhs = valStack.top(); valStack.pop();
                auto ret = applyUnaryOp((UnaryOp)op.second, lhs);
                valStack.push(ret);

            } else if (op.first == Opcode::BINARY_OP) {
                auto lhs = valStack.top(); valStack.pop();
                auto rhs = valStack.top(); valStack.pop();
                auto ret = applyBinaryOp((BinaryOp)op.second, lhs, rhs);
                valStack.push(ret);

            }
        }
    }
};



int main(void)
{
    Primitive p;
    Statement s;

    s.mAttrNameToIndex["pos"] = 0;
    s.mAttrNameToIndex["vel"] = 1;
    p.mAttrValues.emplace_back(1.0);
    p.mAttrValues.emplace_back(2.1);

    s.parse("@pos @vel +add =pos");  // @pos = @pos + @vel
    s.execute(&p);
    std::cout << "pos = " << p.at(0) << std::endl;
    std::cout << "vel = " << p.at(1) << std::endl;

    return 0;
}
