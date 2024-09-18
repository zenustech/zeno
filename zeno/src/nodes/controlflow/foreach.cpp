#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>
#include <zeno/core/Graph.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/core/reflectdef.h>
#include <zeno/formula/zfxexecute.h>
#include <zeno/core/FunctionManager.h>
#include <zeno/types/GeometryObject.h>
#include "zeno_types/reflect/reflection.generated.hpp"


namespace zeno
{
    struct ZDEFNODE() ForEachBegin : INode
    {
        friend struct ForEachEnd;

        ReflectCustomUI m_uilayout = {
            _ObjectGroup {
                {
                    _ObjectParam {"init_object", "Initial Object", Socket_Clone},
                }
            },
            //以下填的是以参数形式返回的外部引用
            _ObjectGroup {
                {
                    //空字符串默认mapping到 apply的输出值
                }
            },
            //返回值信息：
            _ObjectParam {
                "", "Output Object", Socket_Output
            },
            _ParamTab {
            },
            _ParamGroup {
            }
        };


        ZPROPERTY(Role = zeno::Role_InputPrimitive, DisplayName = "Fetch Method", Control = zeno::Combobox, ComboBoxItems = ("Initial Object", "From Last Feedback", "Element of Object"))
        std::string m_fetch_mehod;

        ZPROPERTY(Role = zeno::Role_InputPrimitive, DisplayName = "ForEachEnd Path")
        std::string m_foreach_end_path;

        ZPROPERTY(Role = zeno::Role_InputPrimitive, DisplayName = "Current Iteration")
        int m_current_iteration = 0;

        ZPROPERTY(Role = zeno::Role_OutputPrimitive, DisplayName = "Current Iteration")
        int _out_iteration = 0;


        std::shared_ptr<IObject> apply(std::shared_ptr<IObject> init_object) {
            _out_iteration = m_current_iteration;
            if (m_fetch_mehod == "Initial Object") {
                return init_object;
            }
            else if (m_fetch_mehod == "From Last Feedback") {
                std::shared_ptr<Graph> graph = this->getGraph().lock();
                std::shared_ptr<INode> foreach_end = graph->getNode(m_foreach_end_path);
                if (!foreach_end) {
                    throw makeError<KeyError>("foreach_end_path", "the path of foreach_end_path is not exist");
                }
                int startValue = zeno::reflect::any_cast<int>(foreach_end->get_defl_value("Start Value"));
                if (startValue == m_current_iteration) {
                    return init_object;
                }
                else {
                    std::shared_ptr<IObject> outputObj = foreach_end->get_output_obj("Output Object");
                    //outputObj of last iteration as a feedback to next procedure.
                    return outputObj;
                }
            }
            else if (m_fetch_mehod == "Element of Object") {
                //TODO
                return nullptr;
            }
            return nullptr;
        }
    };

    struct ZDEFNODE() ForEachEnd : INode
    {
        ReflectCustomUI m_uilayout = {
            _ObjectGroup {
                {
                    _ObjectParam {"iterate_object", "Iterate Object", Socket_Clone},
                }
            },
            //以下填的是以参数形式返回的外部引用
            _ObjectGroup {
                {
                    //空字符串默认mapping到 apply的输出值
                }
            },
            //返回值信息：
            _ObjectParam {
                "", "Output Object", Socket_Output
            },
            _ParamTab {
                "Tab1",
                {
                    _ParamGroup {
                        "Group1",
                        {
                            _Param { "iterate_method", "Iterate Method", "By Count", "", Combobox, false,
                                        std::vector<std::string>{"By Count", "By Object Container"}},
                            _Param { "collect_method", "Collect Method", "Feedback to Begin", "", Combobox, false,
                                        std::vector<std::string>{"Feedback to Begin", "Gather Each Iteration"}},
                            _Param { "iterations", "Iterations", 10 },
                            _Param { "start_value", "Start Value", 0 },
                            _Param { "increment", "Increment", 1},
                            _Param { "foreach_begin_path", "ForEachBegin Path", "" },
                            _Param { "stop_condition", "Stop Condition", 0 },
                        }
                    },
                }
            },
            _ParamGroup {
            }
        };

        void update_func() const {

        }


        std::shared_ptr<IObject> apply(
            std::shared_ptr<IObject> iterate_object,
            std::string iterate_method,    /*1.iterate_by_count   2.iterate_by_geometry(TODO) */
            std::string collect_method,    /*1.feedback_to_next_begin  2.gather all iterated result*/
            int iterations,
            int start_value,
            int increment,
            std::string foreach_begin_path,
            int stop_condition
        ) {
            std::shared_ptr<IObject> res;

            std::shared_ptr<Graph> graph = this->getGraph().lock();
            std::shared_ptr<ForEachBegin> foreach_begin = std::dynamic_pointer_cast<ForEachBegin>(graph->getNode(foreach_begin_path));
            if (!foreach_begin) {
                throw makeError<KeyError>("foreach_begin_path", "the path of foreach_begin_path is not exist");
            }

            int curr_iter = zeno::reflect::any_cast<int>(foreach_begin->get_defl_value("Current Iteration"));
            if (curr_iter >= iterations) {
                //TODO: stop_condition
                return iterate_object;
            }

            //construct the `result` object
            if (iterate_method == "By Count") {
                if (collect_method == "Feedback to Begin") {
                    //当前更新操作肯定在scope里，不需要防止提交事务
                    foreach_begin->m_current_iteration = curr_iter + increment;
                    graph->applyNode(foreach_begin_path);

                }
                else if (collect_method == "Gather Each Iteration") {

                }
                else {
                    assert(false);
                    return nullptr;
                }
            }
            else {
                //by object container
            }


            return res;
        }
    };
}