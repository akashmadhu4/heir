#ifndef VN_OPS_H
#define VN_OPS_H


#include "lib/Dialect/VN/IR/VNDialect.h"
#include "mlir/include/mlir/IR/OpDefinition.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"

#include "lib/Dialect/VN/IR/VNTypes.h"


#define GET_OP_CLASSES
#include "lib/Dialect/VN/IR/VNOps.h.inc"

#endif