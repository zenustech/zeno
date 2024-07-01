#include <zeno/formula/syntax_tree.h>
#include <zeno/utils/log.h>

namespace zeno {

    template <class T>
    static T get_zfxvar(zfxvariant value) {
        return std::visit([](auto const& val) -> T {
            using V = std::decay_t<decltype(val)>;
            if constexpr (!std::is_constructible_v<T, V>) {
                throw makeError<TypeError>(typeid(T), typeid(V), "get<zfxvariant>");
            }
            else {
                return T(val);
            }
        }, value);
    }

    std::string getOperatorString(nodeType type, operatorVals op)
    {
        if (type == FUNC)
            return "FUNC";

        switch (type)
        {
        case FUNC: return "FUNC";
        case NUMBER: return "NUMBER";
        case STRING: return "STRING";
        case ZENVAR:
        {
            std::string var = "ZENVAR";
            switch (op) {
            case Indexing: var += " [Indexing]"; break;
            case BulitInVar: var += " [$]"; break;
            case AutoIncreaseFirst: var += " ++var"; break;
            case AutoIncreaseLast: var += " var++"; break;
            case AutoDecreaseFirst: var += " --var;"; break;
            case AutoDecreaseLast: var += " var--"; break;
            }
            return var;
        }
        case JUMP: {
            switch (op) {
            case JUMP_BREAK: return "Break";
            case JUMP_CONTINUE: return "Continue";
            case JUMP_RETURN: return "Return";
            }
        }
        case PLACEHOLDER: return "PLACEHOLDER";
        case FOUROPERATIONS: return "OP";
        case COMPOP: return "COMPARE";
        case CONDEXP: return "CONDITION-EXP";
        case ARRAY: return "ARRAY";
        case DECLARE: return "DECLARE";
        case CODEBLOCK: return "CODEBLOCK";
        case IF: return "IF";
        case FOR: return "FOR";
        case FOREACH: return "FOREACH";
        case WHILE: return "WHILE";
        case DOWHILE: return "DO-WHILE";
        case ASSIGNMENT: {
            std::string var = "ASSIGN";
            switch (op) {
            case AssignTo: var += " ="; break;
            case AddAssign: var += " +="; break;
            case MulAssign: var += " *="; break;
            case SubAssign: var += " -="; break;
            case DivAssign: var += " /="; break;
            }
            return var;
        }
        case VARIABLETYPE: {
            switch (op) {
            case TYPE_INT:      return "INT";
            case TYPE_STRING:   return "STRING";
            case TYPE_FLOAT:    return "FLOAT";
            case TYPE_INT_ARR:  return "INT[]";
            case TYPE_FLOAT_ARR:return "FLOAT[]";
            default:
                return "TYPE";
            }
        }
        default:
            break;
        }

        switch (op) {
        case PLUS:
            return "plus";
        case MINUS:
            return "minus";
        case MUL:
            return "mul";
        case DIV:
            return "div";
        case OR:
            return "or";
        case AND:
            return "and";
        case SIN:
            return "sin";
        case SINH:
            return "sinh";
        case COS:
            return "cos";
        case COSH:
            return "cosh";
        case ABS:
            return "abs";
        case DEFAULT_FUNCVAL:
            return "";
        case UNDEFINE_OP:
            return "undefinedOp";
        default:
            return "";
        }
    }

    operatorVals funcName2Enum(std::string func)
    {
        if (func == "sin") {
            return SIN;
        }
        else if (func == "sinh") {
            return SINH;
        }
        else if (func == "cos") {
            return COS;
        }
        else if (func == "cosh") {
            return COSH;
        }
        else if (func == "abs") {
            return ABS;
        }
        return UNDEFINE_OP;
    }

    void addChild(std::shared_ptr<ZfxASTNode> spNode, std::shared_ptr<ZfxASTNode> spChild) {
        spNode->children.insert(spNode->children.begin(), spChild);
        spChild->parent = spNode;
    }

    void appendChild(std::shared_ptr<ZfxASTNode> spNode, std::shared_ptr<ZfxASTNode> spChild) {
        spNode->children.push_back(spChild);
        spChild->parent = spNode;
    }

    std::shared_ptr<ZfxASTNode> newNode(nodeType type, operatorVals op, std::vector<std::shared_ptr<ZfxASTNode>> Children) {
        std::shared_ptr<ZfxASTNode> n = std::make_shared<ZfxASTNode>();
        if (!n)
        {
            exit(0);
        }
        n->type = type;
        n->opVal = op;
        n->value = 0;

        switch (op) {
        case PLUS:
            n->value = "+";
            break;
        case MINUS:
            n->value = "-";
            break;
        case MUL:
            n->value = "*";
            break;
        case DIV:
            n->value = "/";
            break;
        case AND:
            n->value = "&&";
            break;
        case OR:
            n->value = "||";
            break;
        case DEFAULT_FUNCVAL:
            n->value = "";
            break;
        case UNDEFINE_OP:
            n->value = "undefinedOp";
            break;
        }

        n->children = Children;
        n->isParenthesisNode = false;
        n->isParenthesisNodeComplete = false;
        for (auto& child : n->children)
        {
            if (child) {
                child->parent = n;
            }
        }
        return n;
    }

    std::shared_ptr<ZfxASTNode> newNumberNode(float value) {
        std::shared_ptr<ZfxASTNode> n = std::make_shared<ZfxASTNode>();
        if (!n)
        {
            exit(0);
        }
        n->type = NUMBER;
        n->opVal = UNDEFINE_OP;
        n->value = value;
        return n;
    }

    template<typename ... Args>
    std::string string_format(const std::string& format, Args ... args)
    {
        int size_s = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
        if (size_s <= 0) { throw std::runtime_error("Error during formatting."); }
        auto size = static_cast<size_t>(size_s);
        std::unique_ptr<char[]> buf(new char[size]);
        std::snprintf(buf.get(), size, format.c_str(), args ...);
        return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
    }

    void printSyntaxTree(std::shared_ptr<ZfxASTNode> root, std::string original_code)
    {
        std::string printContent = "\noriginal code: " + original_code + '\n';
        if (!root) {
            printContent += "parser failed";
        }
        else {
            print_syntax_tree(root, 0, printContent);
        }
        zeno::log_info(printContent);
    }

    void print_syntax_tree(std::shared_ptr<ZfxASTNode> root, int depth, std::string& printContent) {
        const auto& printVal = [&printContent](std::shared_ptr<ZfxASTNode> root, char* prefix) {
            if (std::holds_alternative<float>(root->value)) {
                printContent += string_format("%f ", std::get<float>(root->value));
            }
            else if (std::holds_alternative<int>(root->value)) {
                printContent += string_format("%d ", std::get<int>(root->value));
            }
            else if (std::holds_alternative<std::string>(root->value)) {
                printContent += std::get<std::string>(root->value) + " ";
            }
            printContent += "[" + getOperatorString(root->type, root->opVal) + "]\n";
        };

        if (root) {
            for (int i = 0; i < depth; ++i) {
                printContent += "|  ";
            }
            if (std::shared_ptr<ZfxASTNode> spParent = root->parent.lock())
            {
                for (auto& child : spParent->children)
                {
                    if (child)
                    {
                    }
                }
                for (int i = 0; i < spParent->children.size(); i++)
                {
                    if (spParent->children[i] && spParent->children[i] == root) {
                        std::string info = "child:" + std::to_string(i);
                        printVal(root, info.data());
                    }
                }
            }
            else {
                printVal(root, "root");
            }
            for (auto& child : root->children) {
                print_syntax_tree(child, depth + 1, printContent);
            }
        }
    }

    float calc_syntax_tree(std::shared_ptr<ZfxASTNode> root)
    {
        if (root) {
            if (root->type == NUMBER) {
                return std::get<float>(root->value);
            }
            else if (root->type == FOUROPERATIONS) {
                float leftRes = calc_syntax_tree(root->children[0]);
                float rightRes = calc_syntax_tree(root->children[1]);
                switch (root->opVal)
                {
                case PLUS:
                    return leftRes + rightRes;
                case MINUS:
                    return leftRes - rightRes;
                case MUL:
                    return leftRes * rightRes;
                case DIV: {
                    if (rightRes == 0) {
                        return 0;
                    }
                    return leftRes / rightRes;
                }
                default:
                    return 0;
                }
            }
            else if (root->type == FUNC) {
                //TODO: 多元函数的情况
                float leftRes = calc_syntax_tree(root->children[0]);
                switch (root->opVal)
                {
                case SIN:
                    return std::sin(leftRes);
                case SINH:
                    return std::sinh(leftRes);
                case COS:
                    return std::cos(leftRes);
                case COSH:
                    return std::cosh(leftRes);
                case ABS:
                    return std::abs(leftRes);
                default:
                    return 0;
                }
            }
        }
        return 0;
    }

    void currFuncNamePos(std::shared_ptr<ZfxASTNode> root, std::string& name, int& pos)
    {
        std::vector<std::shared_ptr<ZfxASTNode>> preorderVec;
        preOrderVec(root, preorderVec);
        if (preorderVec.size() != 0)
        {
            auto last = preorderVec.back();
            if (last->type == FUNC) {
                std::string candidate = getOperatorString(last->type, last->opVal);
                if (candidate != "") {
                    name = candidate;
                    pos = NAN;  //当前焦点在函数名上
                    return;
                }
            }
            last = last->parent.lock();
            while (last)
            {
                if (last->type == FUNC && !last->isParenthesisNodeComplete) {
                    std::string candidate = getOperatorString(last->type, last->opVal);
                    if (candidate != "") {
                        name = candidate;
                        pos = last->children.size() - 1;
                        return;
                    }
                }
                last = last->parent.lock();
            }
        }
        name = "";
    }

    void preOrderVec(std::shared_ptr<ZfxASTNode> root, std::vector<std::shared_ptr<ZfxASTNode>>& tmplist)
    {
        if (root)
        {
            tmplist.push_back(root);
            for (auto& child : root->children) {
                preOrderVec(child, tmplist);
            }
        }
    }

    void removeAstNode(std::shared_ptr<ZfxASTNode> root) {
        auto parent = root->parent.lock();
        if (!parent) return;
        parent->children.erase(std::remove(parent->children.begin(), parent->children.end(), root)
            , parent->children.end());
    }

    int markOrder(std::shared_ptr<ZfxASTNode> root, int startIndex) {
        for (auto spChild : root->children) {
            startIndex = markOrder(spChild, startIndex);
        }
        root->sortOrderNum = startIndex + 1;
        return startIndex + 1;
    }

    void findAllZenVar(std::shared_ptr<ZfxASTNode> root, std::set<std::string>& vars) {
        if (!root)
            return;
        for (auto spChild : root->children) {
            if (spChild->type == ZENVAR) {
                if (std::holds_alternative<std::string>(spChild->value)) {
                    std::string varname = std::get<std::string>(spChild->value);
                    vars.insert(varname);
                }
            }
            else {
                //找依赖变量只是为了找属性依赖，不能跨越block的范围。
                if (spChild->type != CODEBLOCK)
                    findAllZenVar(spChild, vars);
            }
        }
    }

    std::string decompile(std::shared_ptr<ZfxASTNode> root, const std::string& indent) {
        if (!root) {
            return "";
        }
        switch (root->type)
        {
        case BOOLTYPE:
        case NUMBER: {
            return indent + std::visit([](auto const& val) -> std::string {
                using V = std::decay_t<decltype(val)>;
                if constexpr (std::is_same_v<int, V> || std::is_same_v<float, V>) {
                    return std::to_string(val);
                }
                else {
                    throw makeError("ERROR TYPE OF NUMBER");
                }
            }, root->value);
        }
        case STRING: {
            std::string res = indent + '"' + get_zfxvar<std::string>(root->value) + '"';
            return res;
        }
        case ARRAY: {
            std::string res = "ARRAY";

            return res;
        }
        case ZENVAR: {
            std::string varname = get_zfxvar<std::string>(root->value);
            if (root->opVal == COMPVISIT) {
                if (root->children.size() == 1) {
                    varname = varname + "." + get_zfxvar<std::string>(root->children[0]->value);
                }
            }
            else if (root->opVal == Indexing) {
                if (root->children.size() == 1) {
                    varname = varname + "[" + decompile(root->children[0]) + "]";
                }
            }
            else if (root->opVal == BulitInVar) {
                varname = "$" + varname;
            }
            else if (root->opVal == AutoIncreaseFirst) {
                varname = "++" + varname;
            }
            else if (root->opVal == AutoIncreaseLast) {
                varname = varname + "++";
            }
            else if (root->opVal == AutoDecreaseFirst) {
                varname = "--" + varname;
            }
            else if (root->opVal == AutoDecreaseLast) {
                varname = varname + "--";
            }
            return indent + varname;
        }
        case DECLARE: {
            int N = root->children.size();
            if (N < 2) {
                throw makeError("ERROR ARGS NUMBER OF DECLARE");
            }
            auto spType = root->children[0];
            auto spVarName = root->children[1];
            std::string res;
            res = get_zfxvar<std::string>(spType->value);
            res += " ";
            res += get_zfxvar<std::string>(spVarName->value);
            if (N == 3) {
                auto spValueNode = root->children[2];
                res += " = " + decompile(spValueNode);
            }
            res += ";";
            return indent + res;
        }
        case ASSIGNMENT: {
            if (root->children.size() != 2) {
                throw makeError("ERROR ARGS NUMBER OF ASSIGNMENT");
            }
            auto varNode = root->children[0];
            auto valueNode = root->children[1];
            std::string res;
            res = decompile(varNode);
            switch (root->opVal)
            {
            case AssignTo: res += " = "; break;
            case AddAssign: res += " += "; break;
            case MulAssign: res += " *= "; break;
            case DivAssign: res += " /= "; break;
            default:
                res += " what? "; break;
            }
            res += decompile(valueNode);
            res += ";";
            return indent + res;
        }
        case FUNC:
        {
            std::string funcname = get_zfxvar<std::string>(root->value);
            std::string res = funcname;
            res += "(";
            for (int i = 0; i < root->children.size(); i++) {
                auto spChild = root->children[i];
                res += decompile(spChild);
                if (i < root->children.size() - 1)
                    res += ",";
                else
                    res += "";
            }
            res += ")";
            return indent + res;
        }
        case FOUROPERATIONS:
        case COMPOP:
        {
            std::string res;
            if (root->children.size() != 2) {
                throw makeError("ERROR ARGS NUMBER OF OP");
            }

            std::string op;
            switch (root->opVal)
            {
            case Less:  op = "<"; break;
            case LessEqual: op = "<="; break;
            case Greater:   op = ">";   break;
            case GreaterEqual:  op = ">=";  break;
            case Equal: op = "=="; break;
            case NotEqual:  op = "!="; break;
            case PLUS: op = "+"; break;
            case MINUS: op = "-"; break;
            case MUL: op = "*"; break;
            case DIV: op = "/"; break;
            case AND: op = "&&"; break;
            case OR: op = "||"; break;
            default:
                op = "what?";
            }

            res = decompile(root->children[0]) + op + decompile(root->children[1]);
            return indent + res;
        }
        case CONDEXP:
        {
            if (root->children.size() != 3) {
                throw makeError("ERROR ARGS NUMBER OF CONDEXP");
            }
            auto spCond = root->children[0];
            auto yesStmt = root->children[1];
            auto noStmt = root->children[2];
            std::string res;
            res = "(" + decompile(spCond) + ") ? " + decompile(yesStmt) + " : " + decompile(noStmt) + ";";
            return indent + res;
        }
        case IF:
        {
            std::string res;
            if (root->children.size() != 2) {
                throw makeError("ERROR ARGS NUMBER OF IF");
            }
            res = "if (" + decompile(root->children[0]) + ")\n" +
                decompile(root->children[1], indent);
            return indent + res;
        }
        case WHILE:
        {
            std::string res;
            if (root->children.size() != 2) {
                throw makeError("ERROR ARGS NUMBER OF IF");
            }
            res = "while (" + decompile(root->children[0]) + ")\n" +
                decompile(root->children[1], indent);
            return indent + res;
        }
        case DOWHILE:
        {
            std::string res;
            if (root->children.size() != 2) {
                throw makeError("ERROR ARGS NUMBER OF IF");
            }
            res = "do\n" + decompile(root->children[1], indent) + "while (" + decompile(root->children[0]) + ")\n";
            return indent + res;
        }
        case FOR:
        {
            std::string res;
            res = "for(" + decompile(root->children[0]) + "; " + decompile(root->children[1]) + "; "
                + decompile(root->children[2]) + ")\n" + decompile(root->children[3], indent);
            return indent + res;
        }
        case FOREACH:
        {
            std::string res = "foreach(";
            auto foreachCode = root->children.back();
            auto& children = root->children;
            if (children.size() == 4) {
                res += "[" + get_zfxvar<std::string>(children[0]->value) + "," + get_zfxvar<std::string>(children[1]->value) + "]";
                res += " : " + get_zfxvar<std::string>(children[2]->value) + ")";
            }
            else{
                res += get_zfxvar<std::string>(children[0]->value);
                res += " : " + get_zfxvar<std::string>(children[1]->value) + ")";
            }
            res += decompile(foreachCode, indent);
            return indent + res;
        }
        case FOREACH_ATTR:
        {
            std::string res = "foreach_attr([";
            std::vector<std::string> attrs;
            for (int i = 0; i < root->children[0]->children.size(); i++)
            {
                auto spVar = root->children[0]->children[i];
                res += (get_zfxvar<std::string>(spVar->value));
                if (i < root->children[0]->children.size() - 1)
                    res += ", ";
                else
                    res += "";
            }
            res += "] : prim)\n";
            res += decompile(root->children[1], indent);
            return indent + res;
        }
        case CODEBLOCK:
        {
            std::string res;
            res = indent + "{\n";
            for (auto spChild : root->children) {
                res += decompile(spChild, indent + "    ") + "\n";
            }
            res += indent + "}";
            return res;
        }
        case JUMP:
        {
            std::string op;
            switch (root->opVal)
            {
            case JUMP_RETURN:  op = "return"; break;
            case JUMP_CONTINUE: op = "continue"; break;
            case JUMP_BREAK:   op = "break";   break;
            default:
                op = "what?";
            }
            return indent + op;
        }
        default:
        {

        }
        }
        return "";
    }

    std::shared_ptr<ZfxASTNode> clone(std::shared_ptr<ZfxASTNode> spNode) {
        if (!spNode)
            return nullptr;
        std::shared_ptr<ZfxASTNode> spCloned = std::make_shared<ZfxASTNode>();
        spCloned->code = spNode->code;
        spCloned->value = spNode->value;
        spCloned->bAttr = spNode->bAttr;
        spCloned->AttrAssociateVar = false;
        spCloned->sortOrderNum = spNode->sortOrderNum;
        spCloned->type = spNode->type;
        spCloned->opVal = spNode->opVal;
        for (auto spChild : spNode->children) {
            auto spClonedChild = clone(spChild);
            spCloned->children.push_back(spClonedChild);
            spClonedChild->parent = spCloned;
        }
        return spCloned;
    }

    bool checkparentheses(std::string& exp, int& addleft, int& addright)
    {
        int left = 0;
        int leftNeed = 0;
        for (auto& i : exp)
        {
            if (i == '(')
                left++;
            else if (i == ')')
                left--;
            if (left < 0)
            {
                leftNeed++;
                left = 0;
            }
        }
        if (left != 0 || leftNeed != 0)
        {
            exp = exp + std::string(left, ')');
            exp = std::string(leftNeed, '(') + exp;
            addleft = leftNeed;
            addright = left;
            return false;
        }
        else {
            addleft = 0;
            addright = 0;
            return true;
        }
    }
}