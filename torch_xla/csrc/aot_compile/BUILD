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
        "@xla//xla/python/pjrt_ifrt:pjrt_ifrt",
        "@xla//xla/client:client_library",
        "@xla//xla/service:service",
        "@xla//xla/stream_executor:stream_executor",
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