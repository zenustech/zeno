#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/format.h>
#include <zeno/utils/fileio.h>
#include <zeno/types/ListObject.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/types/ListObject.h>
#include <zeno/utils/string.h>
#include <zeno/utils/logger.h>
#include <string_view>
#include <regex>
#include <charconv>

namespace zeno {
namespace {

struct MakeWritePath : zeno::INode {
    virtual void apply() override {
        auto obj = std::make_unique<zeno::StringObject>();
        obj->set(get_param<std::string>("path"));
        set_output("path", std::move(obj));
    }
};

ZENDEFNODE(MakeWritePath, {
    {},
    {{"string", "path"}},
    {{"writepath", "path", ""}},
    {"string"},
});

struct MakeReadPath : zeno::INode {
    virtual void apply() override {
        auto obj = std::make_unique<zeno::StringObject>();
        obj->set(get_param<std::string>("path"));
        set_output("path", std::move(obj));
    }
};

ZENDEFNODE(MakeReadPath, {
    {},
    {{"string", "path"}},
    {{"readpath", "path", ""}},
    {"string"},
});

struct MakeString : zeno::INode {
    virtual void apply() override {
        auto obj = std::make_unique<zeno::StringObject>();
        obj->set(get_param<std::string>("value"));
        set_output("value", std::move(obj));
    }
};

ZENDEFNODE(MakeString, {
    {},
    {{"string", "value"}},
    {{"string", "value", ""}},
    {"string"},
});

struct MakeMultilineString : MakeString {
};

ZENDEFNODE(MakeMultilineString, {
    {},
    {{"string", "value"}},
    {{"multiline_string", "value", ""}},
    {"string"},
});

struct StringEqual : zeno::INode {
    virtual void apply() override {
        auto lhs = get_input2<std::string>("lhs");
        auto rhs = get_input2<std::string>("rhs");
        set_output2("isEqual", lhs == rhs);
    }
};

ZENDEFNODE(StringEqual, {
    {{"string", "lhs"}, {"string", "rhs"}},
    {{"bool", "isEqual"}},
    {},
    {"string"},
});

struct PrintString : zeno::INode {
    virtual void apply() override {
        auto str = get_input2<std::string>("str");
        printf("PrintString: %s\n", str.c_str());
    }
};

ZENDEFNODE(PrintString, {
    {{"string", "str"}},
    {},
    {},
    {"string"},
});

struct FileWriteString
    : zeno::INode
{
    virtual void apply() override
    {
        auto path = get_input<zeno::StringObject>("path")->get();
        auto str = get_input<zeno::StringObject>("str")->get();
        zeno::file_put_content(path, str);
    }
};

ZENDEFNODE(
    FileWriteString,
    {
        {
            {"string", "str", ""},
            {"writepath", "path", ""},
        },
        {},
        {},
        {"string"},
    });

struct FileReadString
    : zeno::INode
{
    virtual void apply() override
    {
        auto path = get_input<zeno::StringObject>("path")->get();
        auto str = zeno::file_get_content(path);
        set_output2("str", std::move(str));
    }
};

ZENDEFNODE(
    FileReadString,
    {
        {
            {"readpath", "path"},
        },
        {
            {"string", "str"},
        },
        {},
        {"string"},
    });

struct StringFormat : zeno::INode {
    virtual void apply() override {
        auto str = get_input2<std::string>("str");
        for (int i = 0; i < str.size() - 1; i++) {
            if (str[i] == '$' && str[i + 1] == 'F') {
                str.replace(i, 2, std::to_string(getGlobalState()->frameid));
                break;
            }
        }
        set_output2("str", str);
    }
};

ZENDEFNODE(StringFormat, {
    {{"string", "str"}},
    {{"string", "str"}},
    {},
    {"deprecated"},
});

struct StringFormatNumber : zeno::INode {
    virtual void apply() override {
        auto str = get_input2<std::string>("str");
        auto num = get_input<zeno::NumericObject>("number");

        std::string output;
        std::visit([&](const auto &v) {
            //using T = std::decay_t<decltype(v)>;
            //if constexpr (std::is_same_v<T, int>) {
                //output = zeno::format(str, T(v));
                output = zeno::format(str, v);
            //}
            //else if constexpr (std::is_same_v<T, float>) {
                //output = zeno::format(str, T(v));
            //}
            //else {
                //output = str;
            //}
        }, num->value);
        set_output2("str", output);
    }
};

ZENDEFNODE(StringFormatNumber, {
    {
        {"string", "str", "{}"},
        {"number"},
    },
    {{"string", "str"}},
    {},
    {"deprecated"},
});

struct StringFormatNumStr : zeno::INode {
    virtual void apply() override {
        auto str = get_input2<std::string>("str");
        auto num_str = get_input<zeno::IObject>("num_str");
        std::string output;

        std::shared_ptr<zeno::NumericObject> num = std::dynamic_pointer_cast<zeno::NumericObject>(num_str);
        if (num) {
            std::visit([&](const auto &v) {
                output = zeno::format(str, v);
            }, num->value);
        }
        std::shared_ptr<zeno::StringObject> pStr = std::dynamic_pointer_cast<zeno::StringObject>(num_str);
        if (pStr) {
            output = zeno::format(str, pStr->get());
        }
        set_output2("str", output);
    }
};

ZENDEFNODE(StringFormatNumStr, {
    {
        {"string", "str", "{}"},
        {"num_str"},
    },
    {{"string", "str"}},
    {},
    {"string"},
});

struct StringRegexMatch : zeno::INode {
    virtual void apply() override {
        auto str = get_input2<std::string>("str");
        auto regex_str = get_input2<std::string>("regex");
        std::regex self_regex(regex_str);
        int output = std::regex_match(str, self_regex);

        set_output2("output", output);
    }
};

ZENDEFNODE(StringRegexMatch, {
    {
        {"string", "str", ""},
        {"string", "regex", ""},
    },
    {
        {"int", "output"}
    },
    {},
    {"string"},
});

struct StringSplitAndMerge: zeno::INode{
    
    std::vector<std::string> split(const std::string& s, std::string seperator)
    {
        std::vector<std::string> output;

        std::string::size_type prev_pos = 0, pos = 0;

        while((pos = s.find(seperator, pos)) != std::string::npos)
        {
            std::string substring( s.substr(prev_pos, pos-prev_pos) );

            output.push_back(substring);

            prev_pos = ++pos;
        }

        output.push_back(s.substr(prev_pos, pos-prev_pos)); // Last word

        return output;
    }
    virtual void apply() override {
        auto str = get_input2<std::string>("str");
        auto schar = get_input2<std::string>("schar");
        auto merge = get_input2<std::string>("merge");
        
        const auto &strings = split(str, schar);
        const auto &merges = split(merge, ",");
        std::string outputstr = "";
        for(auto idx:merges)
        {
            outputstr += strings[std::atoi(idx.c_str())];
        }

        set_output2("output", outputstr);
    }
};

ZENDEFNODE(StringSplitAndMerge, {
    {
        {"string", "str", ""},
        {"string", "schar", "_"},
        {"string", "merge", "0"},
    },
    {
        {"string", "output"}
    },
    {},
    {"deprecated"},
});

std::vector<std::string_view> stringsplit(std::string_view str, std::string_view delims = " ")//do not keep empty
{
	std::vector<std::string_view> output;
        size_t pos = 0;
        size_t posbegin = 0;
        std::string_view word;
        while ((pos = str.find(delims, pos)) != std::string_view::npos) {
            word = str.substr(posbegin, pos-posbegin);
            output.push_back(word);
            pos += delims.length();
            posbegin = pos;
        }
        if (posbegin < str.length()) { 
            word = str.substr(posbegin);
            output.push_back(word);
        }
	return output;
}

struct StringSplitAndMerge2: zeno::INode{
    virtual void apply() override {
        auto str = get_input2<std::string>("String");
        auto separator = get_input2<std::string>("Separator");
        auto mergeMethod = get_input2<std::string>("Merge Method");
        auto mergeIndex = get_input2<std::string>("Merge index");
        auto clipCountFromStart = get_input2<int>("Clip Count From Start");
        auto clipCountFromEnd = get_input2<int>("Clip Count From End");
        auto remainSeparator = get_input2<bool>("Remain Separator");
        auto splitStr = stringsplit(str, separator);
        std::string output;
        output.reserve(str.size());
        if (mergeMethod == "Custom_Index_Merge") {
            std::vector<std::string_view> mergeIndexList = stringsplit(mergeIndex, ",");
            for (size_t j = 0; j < mergeIndexList.size(); ++j) {
                auto idx = mergeIndexList[j];
                if (idx.empty()) continue;
                int i;
                auto result = std::from_chars(idx.data(), idx.data() + idx.size(), i);
                if (result.ec == std::errc::invalid_argument || result.ec == std::errc::result_out_of_range || result.ptr != idx.data() + idx.size()) {
                    throw std::runtime_error("[StringSplitAndMerge2] Merge index is not a valid number.");//invalid_argument, result_out_of_range, or not all characters are parsed(eg. 123a)
                }
                if (i < 0) i = splitStr.size() + i;
                if (i < 0 || i >= splitStr.size()) {
                    throw std::runtime_error("[StringSplitAndMerge2] Merge index is out of range.");
                }
                output += splitStr[i];
                if (remainSeparator && j != mergeIndexList.size() - 1) {
                    output += separator;
                }
            }
        }
        else if (mergeMethod == "Clip_And_Merge") {
            int start = std::max(0, clipCountFromStart);
            int end = std::max(start, static_cast<int>(splitStr.size()) - clipCountFromEnd);
            for (int i = start; i < end; ++i) {
                output += splitStr[i];
                if (remainSeparator && i != end - 1) {
                    output += separator;
                }
            }
        }
        else {
            throw std::runtime_error("[StringSplitAndMerge2] Unknown merge method.");
        }
        set_output2("string", output);
    }
};

ZENDEFNODE(StringSplitAndMerge2, {
    {
        {"multiline_string", "String", ""},
        {"string", "Separator", "_"},
        {"enum Custom_Index_Merge Clip_And_Merge", "Merge Method", "Custom_Index_Merge"},
        {"string", "Merge index", "0,1"},
        {"int", "Clip Count From Start", "0"},
        {"int", "Clip Count From End", "0"},
        {"bool", "Remain Separator", "false"},
    },
    {
        {"string", "string"}
    },
    {},
    {"string"},
});

struct FormatString : zeno::INode {
    virtual void apply() override {
        auto formatStr = get_input2<std::string>("str");

        auto list = get_input<zeno::ListObject>("args");
        std::string output = formatStr;
        for (auto obj : list->arr)
        {
            std::shared_ptr<zeno::NumericObject> num = std::dynamic_pointer_cast<zeno::NumericObject>(obj);
            if (num) {
                std::visit([&](const auto& v) {
                    output = zeno::format(output, v);
                    }, num->value);
            }
            std::shared_ptr<zeno::StringObject> pStr = std::dynamic_pointer_cast<zeno::StringObject>(obj);
            if (pStr) {
                output = zeno::format(output, pStr->get());
            }
        }

        set_output2("str", output);
    }
};

ZENDEFNODE(FormatString, {
    {
        {"string", "str", "{}"},
        {"list", "args"},
    },
    {{"string", "str"}},
    {},
    {"string"},
});


/*static int objid = 0;

struct ExportPath : zeno::INode {  // deprecated
    virtual void apply() override {
        char buf[100];
        auto ext = get_param<std::string>("ext");
        sprintf(buf, "%06d", getGlobalState()->frameid);
        auto path = fs::path(getGlobalState()->iopath) / buf;
        if (!fs::is_directory(path)) {
            fs::create_directory(path);
        }
        sprintf(buf, "%06d.%s", objid++, ext.c_str());
        path /= buf;
        auto ret = std::make_unique<zeno::StringObject>();
        //printf("EXPORTPATH: %s\n", path.c_str());
        ret->set(path.string());
        set_output("path", std::move(ret));
    }
};

ZENDEFNODE(ExportPath, {
    {},
    {"path"},
    {{"string", "ext", "zpm"}},
    {"fileio"},
});

struct EndFrame : zeno::INode {  // deprecated
    virtual void apply() override {
        char buf[100];
        sprintf(buf, "%06d", getGlobalState()->frameid);
        auto path = fs::path(getGlobalState()->iopath) / buf;
        if (!fs::is_directory(path)) {
            fs::create_directory(path);
        }
        path /= "done.lock";
        std::ofstream ofs(path.string());
        ofs.write("DONE", 4);
        objid = 0;
    }
};

ZENDEFNODE(EndFrame, {
    {"chain"},
    {},
    {},
    {"fileio"},
});*/

struct StringToNumber : zeno::INode {
    virtual void apply() override {
        auto in_str = get_input2<std::string>("str");
        auto type = get_input2<std::string>("type");
        auto obj = std::make_unique<zeno::NumericObject>();
        if (type == "float") {
            float v = std::stof(in_str);
            obj->set(v);
        }
        else if (type == "int") {
            int v = std::stoi(in_str);
            obj->set(v);
        }
        else {
            throw zeno::makeError("Unknown type");
        }

        set_output("num_str", std::move(obj));
    }
};

ZENDEFNODE(StringToNumber, {{
                                /* inputs: */
                                {"enum float int", "type", "float"},
                                {"string", "str", "0"},
                            },

                            {
                                /* outputs: */
                                "num_str",
                            },

                            {
                                /* params: */

                            },

                            {
                                /* category: */
                                "string",
                            }});

std::string& trim(std::string &s) 
{
    if (s.empty()) 
    {
        return s;
    }
    s.erase(0,s.find_first_not_of(" \f\n\r\t\v"));
    s.erase(s.find_last_not_of(" \f\n\r\t\v") + 1);
    return s;
}

struct StringToList : zeno::INode {
    virtual void apply() override {
        auto stringlist = get_input2<std::string>("string");
        auto list = std::make_shared<ListObject>();
        auto separator = get_input2<std::string>("Separator");
        auto trimoption = get_input2<bool>("Trim");
        auto keepempty = get_input2<bool>("KeepEmpty");
        std::vector<std::string> strings;
        size_t pos = 0;
        size_t posbegin = 0;
        std::string word;
        while ((pos = stringlist.find(separator, pos)) != std::string::npos) {
            word = stringlist.substr(posbegin, pos-posbegin);
            if(trimoption) trim(word);
            if(keepempty || !word.empty()) strings.push_back(word);
            pos += separator.length();
            posbegin = pos;
        }
        if (posbegin < stringlist.length()) { //push last word
            word = stringlist.substr(posbegin);
            if(trimoption) trim(word);
            if(keepempty || !word.empty()) strings.push_back(word);
        }
        for(const auto &string : strings) {
            auto obj = std::make_unique<StringObject>();
            obj->set(string);
            list->arr.push_back(std::move(obj));
        }
        set_output("list", std::move(list));
    }
};

ZENDEFNODE(StringToList, {
    {
        {"multiline_string", "string", ""},
        {"string", "Separator", ""},
        {"bool", "Trim", "false"},
        {"bool", "KeepEmpty", "false"},
    },
    {{"list"},
    },
    {},
    {"string"},
});

struct StringJoin : zeno::INode {//zeno string only support list for now
    virtual void apply() override {
        auto list = get_input<zeno::ListObject>("list");
        auto stringvec = list->get2<std::string>();
        auto separator = get_input2<std::string>("Separator");
        auto output = join_str(stringvec, separator);
        set_output2("string", output);
    }
};

ZENDEFNODE(StringJoin, {
    {
        {"list"},
        {"string", "Separator", ""},
    },
    {{"string", "string"},
    },
    {},
    {"string"},
});

struct NumbertoString : zeno::INode {
    virtual void apply() override {
        auto num = get_input<zeno::NumericObject>("number");
        auto obj = std::make_unique<zeno::StringObject>();
        std::visit([&](const auto &v) {
            obj->set(zeno::to_string(v));
        }, num->value);
        set_output("string", std::move(obj));
    }
};

ZENDEFNODE(NumbertoString, {
    {
        {"number"},
    },
    {{"string", "string"},
    },
    {},
    {"string"},
});

std::string strreplace(std::string textToSearch, std::string_view toReplace, std::string_view replacement)
{
    size_t pos = 0;
    for (;;)
    {
        pos = textToSearch.find(toReplace, pos);
        if (pos == std::string::npos)
            return textToSearch;
        textToSearch.replace(pos, toReplace.length(), replacement);
        pos += replacement.length();
    }
}

struct StringReplace : zeno::INode {
    virtual void apply() override {
        std::string string = get_input2<std::string>("string");
        std::string oldstr = get_input2<std::string>("old");
        std::string newstr = get_input2<std::string>("new");
        if (oldstr.empty()) {
            zeno::log_error("[StringReplace] old string is empty.");
            return;
        }
        auto output = strreplace(string, oldstr, newstr);
        set_output2("string", output);
    }
};

ZENDEFNODE(StringReplace, {
    {
        {"multiline_string", "string", ""},
        {"string", "old", ""},
        {"string", "new", ""},
    },
    {{"string", "string"},
    },
    {},
    {"string"},
});

struct StringFind : zeno::INode {//return -1 if not found
    virtual void apply() override {
        auto string = get_input2<std::string>("string");
        auto substring = get_input2<std::string>("substring");
        auto start = get_input2<int>("start");
        std::string::size_type n = string.find(substring, start);
        int output = (n == std::string::npos) ? -1 : static_cast<int>(n);
        set_output2("Position", output);
    }
};

ZENDEFNODE(StringFind, {
    {
        {"multiline_string", "string", ""},
        {"string", "substring", ""},
        {"int", "start", "0"},
    },
    {{"int", "Position"},
    },
    {},
    {"string"},
});

struct SubString : zeno::INode {//slice...
    virtual void apply() override {
        auto string = get_input2<std::string>("string");
        auto start = get_input2<int>("start");
        auto length = get_input2<int>("length");
        if (start < 0) {
            start = string.size() + start;
        }
        if (start < 0 || start >= string.size()) {
            throw std::runtime_error("[SubString] start is out of range.");
        }
        auto output = string.substr(start, length);
        set_output2("string", output);
    }
};

ZENDEFNODE(SubString, {
    {
        {"multiline_string", "string", ""},
        {"int", "start", "0"},
        {"int", "length", "1"},
    },
    {{"string", "string"},
    },
    {},
    {"string"},
});

struct StringtoLower : zeno::INode {
    virtual void apply() override {
        auto string = get_input2<std::string>("string");
        std::string output = string;
        std::transform(output.begin(), output.end(), output.begin(), [] (auto c) { 
            return static_cast<char> (std::tolower (static_cast<unsigned char> (c))); });
        set_output2("string", output);
    }
};

ZENDEFNODE(StringtoLower, {
    {
        {"string", "string", ""},
    },
    {{"string", "string"},
    },
    {},
    {"string"},
});

struct StringtoUpper : zeno::INode {
    virtual void apply() override {
        auto string = get_input2<std::string>("string");
        std::string output = string;
        std::transform(output.begin(), output.end(), output.begin(), [] (auto c) { 
            return static_cast<char> (std::toupper (static_cast<unsigned char> (c))); });
        set_output2("string", output);
    }
};

ZENDEFNODE(StringtoUpper, {
    {
        {"string", "string", ""},
    },
    {{"string", "string"},
    },
    {},
    {"string"},
});

struct StringLength : zeno::INode {
    virtual void apply() override {
        auto string = get_input2<std::string>("string");
        int output = string.length();
        set_output2("length", output);
    }
};

ZENDEFNODE(StringLength, {
    {
        {"string", "string", ""},
    },
    {{"int", "length"},
    },
    {},
    {"string"},
});

struct StringSplitPath : zeno::INode {
    virtual void apply() override {
        auto stringpath = get_input2<std::string>("string");
        bool SplitExtension = get_input2<bool>("SplitExtension");
        std::string directory, filename, extension;
        std::string::size_type last_slash_pos = stringpath.find_last_of("/\\");
        std::string::size_type last_dot_pos = stringpath.find_last_of('.');
        if (last_slash_pos == std::string::npos) {
            directory = "";
            filename = (last_dot_pos == std::string::npos) ? stringpath : stringpath.substr(0, last_dot_pos);
            extension = (last_dot_pos == std::string::npos) ? "" : stringpath.substr(last_dot_pos + 1);
        }
        else {
            directory = stringpath.substr(0, last_slash_pos);
            filename = stringpath.substr(last_slash_pos + 1, (last_dot_pos == std::string::npos ? stringpath.length() - last_slash_pos - 1 : last_dot_pos - last_slash_pos - 1));
            extension = (last_dot_pos == std::string::npos) ? "" : stringpath.substr(last_dot_pos + 1);
        }
        if(!SplitExtension) filename += extension;//extension output is empty if SplitExtension is false
        set_output2("directory", directory);
        set_output2("filename", filename);
        set_output2("extension", extension);
    }
};

ZENDEFNODE(StringSplitPath, {
    {
        {"readpath", "string", ""},
        {"bool", "SplitExtension", "true"},
    },
    {{"string", "directory"},
    {"string", "filename"},
    {"string", "extension"},
    },
    {},
    {"string"},
});

struct StringInsert : zeno::INode {//if start is less than 0, reverse counting from the end
    virtual void apply() override {
        auto string = get_input2<std::string>("string");
        auto substring = get_input2<std::string>("substring");
        auto start = get_input2<int>("start");
        auto output = string;
        if (start < 0) {
            start = output.size() + start + 1;
        } 
        if (start < 0 || start > string.size()) {
            throw std::runtime_error("[StringInsert] start is out of range.");
        }
        output.insert(start, substring);
        set_output2("string", output);
    }
};

ZENDEFNODE(StringInsert, {
    {
        {"multiline_string", "string", ""},
        {"string", "substring", ""},
        {"int", "start", "0"},
    },
    {{"string", "string"},
    },
    {},
    {"string"},
});

struct StringTrim : zeno::INode {
    virtual void apply() override {
        auto string = get_input2<std::string>("string");
        auto trimleft = get_input2<bool>("trimleft");
        auto trimright = get_input2<bool>("trimright");
        std::string output = string;
        if (!output.empty()) {
            if (trimleft) {
                output.erase(output.begin(), std::find_if(output.begin(), output.end(), [](int ch) {
                    return !std::isspace(ch);
                }));
            }
            if (trimright) {
                output.erase(std::find_if(output.rbegin(), output.rend(), [](int ch) {
                    return !std::isspace(ch);
                }).base(), output.end());
            }
        }
        set_output2("string", output);
        
    }
};

ZENDEFNODE(StringTrim, {
    {
        {"string", "string", ""},
        {"bool", "trimleft", "true"},
        {"bool", "trimright", "true"},
    },
    {{"string", "string"},
    },
    {},
    {"string"},
});

struct StringDeleteOrReplace : zeno::INode {
    virtual void apply() override {
        std::string multiline_string = get_input2<std::string>("String");
        std::string oldString = get_input2<std::string>("oldString");
        std::string RefString = get_input2<std::string>("RefString");
        auto N = get_input2<int>("N");
        std::string newString = get_input2<std::string>("newString");
        bool UseLastRefString = get_input2<bool>("UseLastRefString");
        std::string output = multiline_string;
        if(oldString == "AllRefString") {
            output = strreplace(multiline_string, RefString, newString);
        }
        else if(oldString == "First_N_characters") {
            if(N >= 0 && N <= multiline_string.size()) {
                output.replace(0, N, newString);
            }
            else {
                //zeno::log_error("[StringDeleteOrReplace] N is out of range.");
                throw std::runtime_error("[StringDeleteOrReplace] N is out of range.");
            }
        }
        else if(oldString == "Last_N_characters") {
            if(N >= 0 && N <= multiline_string.size()) {
                output.replace(multiline_string.size() - N, N, newString);
            }
            else {
                throw std::runtime_error("[StringDeleteOrReplace] N is out of range.");
            }
        }
        else if(oldString == "All_characters_before_RefString") {
            auto pos = UseLastRefString ? multiline_string.rfind(RefString) : multiline_string.find(RefString);
            if(pos != std::string::npos) {
                output.replace(0, pos, newString);
            }
            else {
                throw std::runtime_error("[StringDeleteOrReplace] RefString not found.");
            }
        }
        else if(oldString == "N_characters_before_RefString") {
            auto pos = UseLastRefString ? multiline_string.rfind(RefString) : multiline_string.find(RefString);
            if(pos != std::string::npos && pos >= N) {
                output.replace(pos - N, N, newString);
            }
            else {
                throw std::runtime_error("[StringDeleteOrReplace] RefString not found or N is too large.");
            }
        }
        else if(oldString == "All_characters_after_RefString") {
            auto pos = UseLastRefString ? multiline_string.rfind(RefString) : multiline_string.find(RefString);
            if(pos != std::string::npos) {
                output.replace(pos + RefString.size(), multiline_string.size() - pos - RefString.size(), newString);
            }
            else {
                throw std::runtime_error("[StringDeleteOrReplace] RefString not found.");
            }
        }
        else if(oldString == "N_characters_after_RefString") {
            auto pos = UseLastRefString ? multiline_string.rfind(RefString) : multiline_string.find(RefString);
            if(pos != std::string::npos && pos + RefString.size() + N <= multiline_string.size()) {
                output.replace(pos + RefString.size(), N, newString);
            }
            else {
                throw std::runtime_error("[StringDeleteOrReplace] RefString not found or N is too large.");
            }
        }
        set_output2("string", output);
    }
};

ZENDEFNODE(StringDeleteOrReplace, {
    {
        {"multiline_string", "String", ""},
        {"enum AllRefString First_N_characters  Last_N_characters All_characters_before_RefString  N_characters_before_RefString All_characters_after_RefString N_characters_after_RefString", "oldString", "AllRefString"},
        {"string", "RefString", ""},
        {"bool", "UseLastRefString", "false"},
        {"int", "N", "1"},
        {"string", "newString", ""},
    },
    {{"string", "string"},
    },
    {},
    {"string"},
});

struct StringEditNumber : zeno::INode {
    virtual void apply() override {
        auto string = get_input2<std::string>("String");
        auto method = get_input2<std::string>("Method");
        if (method == "Remove_all_numbers") {
            string.erase(std::remove_if(string.begin(), string.end(), [](char c) { return std::isdigit(c); }), string.end());
        }
        else if (method == "Remove_all_non_numbers") {
            string.erase(std::remove_if(string.begin(), string.end(), [](unsigned char c) { return !std::isdigit(c); }), string.end());
        }
        else if (method == "Remove_last_number") {
            auto it = std::find_if(string.rbegin(), string.rend(), [](unsigned char c) { return std::isdigit(c); });
            if (it != string.rend()) {
            string.erase((it+1).base());
            }
        }
        else if (method == "Return_last_number") {
            std::string num = "";
            bool number_found = false;
            for (auto it = string.rbegin(); it != string.rend(); ++it) {
                if (std::isdigit(*it)) {
                    num = *it + num;
                    number_found = true;
                } else if (number_found) {
                    break;
                }
            }
            string = num;
        }
        set_output2("string", string);
        
    }
};

ZENDEFNODE(StringEditNumber, {
    {
        {"multiline_string", "String", ""},
        {"enum Remove_all_numbers Remove_all_non_numbers Remove_last_number Return_last_number_Sequence", "Method", "Remove_all_numbers"},
    },
    {{"string", "string"},
    },
    {},
    {"string"},
});

}
}
