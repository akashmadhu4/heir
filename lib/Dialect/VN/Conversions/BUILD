load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ConvertLinalgToVN",
    srcs = ["ConvertLinalgToVN.cpp"],
    hdrs = ["ConvertLinalgToVN.h"],
    deps = [
        ":pass_inc_gen",
        "//lib/Dialect/VN/IR:Dialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Pass",
    ],
)
gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            ["-gen-pass-decls"],
            "ConvertLinalgToVN.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ConvertLinalgToVN.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)