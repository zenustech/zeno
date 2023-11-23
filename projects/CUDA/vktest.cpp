#include <shaderc/shaderc.hpp>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_structs.hpp>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/zeno.h>

namespace zeno {

struct TestVk : INode {
    void apply() override {

        const vk::ApplicationInfo aI("Hello World", 0, nullptr, 0, VK_API_VERSION_1_3);
        const auto instance = vk::createInstanceUnique(vk::InstanceCreateInfo({}, &aI));
        const auto physDevice = instance->enumeratePhysicalDevices()[0];

        int family;
        const auto queueProps = physDevice.getQueueFamilyProperties();
        for (family = 0; !(queueProps[family].queueFlags & vk::QueueFlagBits::eCompute) && family < queueProps.size();
             family++)
            ;

        constexpr float priority[] = {1.f};
        const vk::DeviceQueueCreateInfo devQueueCI({}, family, 1, priority);
        const auto device = physDevice.createDeviceUnique(vk::DeviceCreateInfo({}, devQueueCI));

        const std::string print_shader = R"(
#version 460
#extension GL_EXT_debug_printf : require
void main() {
	debugPrintfEXT("hello world! (said thread: %d)\n", gl_GlobalInvocationID.x); 
})";

        const auto compiled =
            shaderc::Compiler().CompileGlslToSpv(print_shader, shaderc_compute_shader, "hello_world.comp");
        const std::vector<uint32_t> spirv(compiled.cbegin(), compiled.cend());
        const auto shaderModule = device->createShaderModuleUnique(vk::ShaderModuleCreateInfo({}, spirv));

        const vk::PipelineShaderStageCreateInfo stageCI({}, vk::ShaderStageFlagBits::eCompute, *shaderModule, "main");
        const auto pipelineLayout = device->createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo());
        // stage (shader module), layout -> pipeline
        const vk::ComputePipelineCreateInfo pipelineCI({}, stageCI, *pipelineLayout);
        const auto [status, pipeline] =
            device->createComputePipelineUnique(*device->createPipelineCacheUnique({}), pipelineCI);

        const auto pool = device->createCommandPoolUnique(vk::CommandPoolCreateInfo({}, family));
        const vk::CommandBufferAllocateInfo allocInfo(*pool, vk::CommandBufferLevel::ePrimary, 1);
        const auto cmdBuffers = device->allocateCommandBuffersUnique(allocInfo);
        cmdBuffers[0]->begin(vk::CommandBufferBeginInfo{});
        cmdBuffers[0]->bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
        cmdBuffers[0]->dispatch(8, 1, 1);
        cmdBuffers[0]->end();

        device->getQueue(family, 0).submit(vk::SubmitInfo({}, {}, *cmdBuffers[0]));
        device->waitIdle();

        return;
    }
};

ZENDEFNODE(TestVk, {{}, {""}, {}, {"vk"}});

} // namespace zeno