#ifndef AOT_COMPILE_PJRT_COMPILER_
#define AOT_COMPILE_PJRT_COMPILER_

#include <torch/torch.h>
#include <ATen/ATen.h>

#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_compiler.h"

#include <string>
#include <memory>

namespace torch_xla {
namespace aot_compile {

/*
 * Simple interface for AOT compile - compiled bits are returned  -
 * i.e. you *just* need to load the compiled string (no recompilation)
 * 
 * Expect to expand this with compile options, load options, topology
 * etc.
 */
class PjRtCompiler {
public:

    PjRtCompiler( const std::string& device_name, const std::string& pjrt_lib_path );
    std::string Compile(const std::string& hlo_proto);

private:

    bool initialize(const std::string& device_name, const std::string& pjrt_lib_path);

    std::string device_name_;
    const PJRT_Api* c_api_;
    std::unique_ptr<xla::PjRtCApiCompiler> compiler_;
};


} // namespace aot_compile
} // namespace torch_xla

#endif // AOT_COMPILE_PJRT_COMPILER_
