cc_library(
    name = "pjrt_c_api_layouts_extension",
    deps = [
        "@xla//xla/pjrt/c:pjrt_c_api_layouts_extension_hdrs",
    ],
)

cc_library(
    name = "pjrt_compile_options",
    deps = [
        "@xla//xla/pjrt:compile_options_proto_cc",
    ],
)

cc_library(
    name = "xla_local_client",
    deps = [
        "@xla//xla/pjrt:pjrt_api",
        "@xla//xla/pjrt:pjrt_c_api_client",
        "//torch_xla/csrc/runtime:runtime",
        "//torch_xla/csrc:tensor",
        "//torch_xla/csrc:aten_cuda_functions",
        "//:_XLAC_cuda_functions.so",
        "@torch//:libc10",
        "@torch//:libtorch",
        "@torch//:libtorch_cpu",      
        "@torch//:libtorch_python"    
    ]
)

cc_library(
    name = "aot_compile",
    hdrs = [
        "aot_compiler.h",
        "aot_model.h"
    ],
    srcs = [
        "aot_compiler.cc",
        "aot_model.cc"
    ],
    deps = [
        ":xla_local_client",
    ],
)

cc_binary(
    name = "neuron_xla_client_test",
    srcs = ["test2.cpp" ],
    deps = [
        ":xla_local_client",
        "@torch//:runtime_headers",
        "@torch//:headers",   
    ],
    includes = [
        "/ansible/src/pytorch/torch/include/torch/csrc/api/include"
    ],
    copts = [
        "-I/usr/local/include/python3.10",
        "-DTORCH_API_INCLUDE_EXTENSION_H",        
    ],
    linkopts = [
        "-L/usr/local/lib","-lpython3.10", "-lutil",
    ],
)

cc_binary(
    name = "neuron_xla_shared_test",
    srcs = ["test.cpp" ],
    deps = [
        ":aot_compile",   
    ],
    copts = [
        "-I/usr/local/include/python3.10",
        "-DTORCH_API_INCLUDE_EXTENSION_H",        
    ],
    linkopts = [
        "-L/usr/local/lib","-lpython3.10", "-lutil",
    ],
)