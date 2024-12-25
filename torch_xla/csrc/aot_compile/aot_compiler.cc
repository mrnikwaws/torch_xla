#include "torch_xla/csrc/aot_compile/aot_compiler.h"

#include "torch_xla/csrc/runtime/tf_logging.h"

namespace torch_xla {
namespace aot_compile {


PjRtCompiler::PjRtCompiler( const std::string& device_name ) 
    :device_name_(device_name), c_api(nullptr), compiler_(nullptr) {
        
    XLA_CHECK( initialize(device_name) );
}

bool PjRtCompiler::initialize( const std::string& device_name, const std::string& pjrt_lib_path ) {
    auto status_or_api = pjrt::PjrtApi("NEURON");
    XLA_CHECK_OK(status_or_api.status());

    if( !status_or_api.value() ) {
        auto status_or_plugin_loaded = pjrt::LoadPjrtPlugin(device_name, pjrt_lib_path)
        XLA_CHECK_OK(status_or_plugin_loaded.status());

        VLOG(1) << "'" << device_name << "' plugin loaded = " << status_or_plugin_loaded.value();

        if( !status_or_plugin_loaded.value() ) {
            return false;
        }

        status_or_c_api = pjrt::PjrtApi(device_name);
        XLA_CHECK_OK(status_or_plugin_loaded.status());

        if( status_or_c_api.value() == nullptr ) {
            return false;
        }
    }

    c_api_ = status_or_c_api.value();

    xla::PjRtCApiCompiler neuron_compiler(c_api_);
    compiler_ = std::move(std::make_unique<xla::PjRtCApiCompiler>(c_api_))

    return true;
}

std::string PjRtCompiler::Compile(
    const std::string_view hlo_proto) {

    auto status_or_topology = xla::GetCApiTopology(device_name_, device_name_);
    XLA_CHECK(status_or_topology.ok());

    std::unique_ptr<xla::PjRtTopologyDescription> topology_desc = std::move(status_or_topology.value());
    xla::XlaComputation comp(hlo_proto);
    xla::CompileOptions options;

    auto executable_status_or = compiler_->Compile(options, comp, *topology_desc, nullptr);
    XLA_CHECK_OK(executable_status_or.status());

    auto executable_shared_ptr = std::move(executable_status_or.value());

    // Serialize the compiled executable
    auto status_or_serial = executable_shared_ptr->SerializeExecutable();

    if( !status_or_serial.ok() ) {
        std::cout << "Serialization of compiled artifacts faield" << std::endl;
        return 1;
    }

    // Serialized compiled bits! Not a cache handle
    std::string serialized_executable = status_or_serial.value();

    // Don't use std::move - return value optimization will take care of this
    // https://stackoverflow.com/questions/12011426/how-to-use-move-semantics-with-stdstring-during-func
    return serialized_executable;   
}

} // namespace aot_compile
} // namespace torch_xla