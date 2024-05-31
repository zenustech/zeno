#include <zeno/formula/syntax_tree.h>

char* getOperatorString(operatorVals op)
{
    switch (op) {
    case PLUS:
        return "plus";
    case MINUS:
        return "minus";
    case MUL:
        return "mul";
    case DIV:
        return "div";
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
        return "default_funcVal";
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
    } else if (func == "sinh") {
        return SINH;
    }else if (func == "cos") {
        return COS;
    }else if (func == "cosh") {
        return COSH;
    } else if (func == "abs") {
        return ABS;
    }
    return UNDEFINE_OP;
}

std::shared_ptr<struct node> newNode(nodeType type, operatorVals op, std::vector<std::shared_ptr<struct node>> Children) {
    std::shared_ptr<struct node> n = std::make_shared<struct node>();
    if (!n)
    {
        exit(0);
    }
    n->type = type;
    n->opVal = op;
    n->value = 0;
    n->children = Children;
    n->isParenthesisNode = false;
    n->isParenthesisNodeComplete = false;
    for (auto & child: n->children)
    {
        if (child) {
            child->parent = n;
        }
    }
    return n;
}

std::shared_ptr<struct node> newNumberNode(float value) {
    std::shared_ptr<struct node> n = std::make_shared<struct node>();
    if (!n)
    {
        exit(0);
    }
    n->type = NUMBER;
    n->opVal = UNDEFINE_OP;
    n->value = value;
    return n;
}

void print_syntax_tree(std::shared_ptr<struct node> root, int depth) {
    const auto& printVal = [](std::shared_ptr<struct node> root, char* prefix) {
        if (root->type == NUMBER)
            printf("%s: %f isParen:%d : iscompleted:%d\n", prefix, root->value, root->isParenthesisNode, root->isParenthesisNodeComplete);
        else
            printf("%s: %s isParen:%d : iscompleted:%d\n", prefix, getOperatorString(root->opVal), root->isParenthesisNode, root->isParenthesisNodeComplete);
    };
    if (root) {
        for (int i = 0; i < depth; ++i) {
            printf("|  ");
        }
        if (std::shared_ptr<struct node> spParent = root->parent.lock())
        {
            for (auto& child: spParent->children)
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
        for (auto& child: root->children) {
            print_syntax_tree(child, depth + 1);
        }
    }
}

float calc_syntax_tree(std::shared_ptr<struct node> root)
{
    if (root) {
        if (root->type == NUMBER)
        {
            return root->value;
        } else if (root->type == FOUROPERATIONS) {
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
        } else if (root->type == FUNC) {
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

void currFuncNamePos(std::shared_ptr<struct node> root, std::string& name, int& pos)
{
    std::vector<std::shared_ptr<struct node>> preorderVec;
    preOrderVec(root, preorderVec);
    if (preorderVec.size() != 0)
    {
        auto last = preorderVec.back();
        if (last->type == FUNC) {
            std::string candidate = getOperatorString(last->opVal);
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
                std::string candidate = getOperatorString(last->opVal);
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

void preOrderVec(std::shared_ptr<struct node> root, std::vector<std::shared_ptr<struct node>>& tmplist)
{
    if (root)
    {
        tmplist.push_back(root);
        for (auto& child: root->children) {
            preOrderVec(child, tmplist);
        }
    }
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