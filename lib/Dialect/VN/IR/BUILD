load("@heir//lib/Dialect:dialect.bzl", "add_heir_dialect_library")
load("@llvm-project//mlir:tblgen.bzl", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)


cc_library(
    name = "Dialect",
    srcs = [
        "VNDialect.cpp",
    ],
    hdrs = [
        "VNDialect.h",
        "VNOps.h",
        "VNTypes.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)


# Collect all .td files
td_library(
    name = "td_files",
    srcs = [
        "VNDialect.td",
        "VNOps.td",
        "VNTypes.td",
    ],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)


add_heir_dialect_library(
    name = "dialect_inc_gen",
    dialect = "VN",
    kind = "dialect",
    td_file = "VNDialect.td",
    deps = [":td_files"],
)

# 3. Generate .inc files for types
add_heir_dialect_library(
    name = "types_inc_gen",
    dialect = "VN",
    kind = "type",
    td_file = "VNTypes.td",
    deps = [":td_files"],
)

# 4. Generate .inc files for ops
add_heir_dialect_library(
    name = "ops_inc_gen",
    dialect = "VN",
    kind = "op",
    td_file = "VNOps.td",
    deps = [":td_files"],
)


