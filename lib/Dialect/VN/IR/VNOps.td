
include "VNTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"





class VN_Op<string mnemonic, list<Trait>traits = []>
    :Op<VN_Dialect,mnemonic,traits>{
        let cppNamespace = "::mlir::heir::vn";
    }



def VN_MulOp : VN_Op<"mul", [Commutative]> {
    let summary = "Elementwise multiplication of data within VN";

    let description = [{
        Multiply two tensors elementwise with broadcasting support
    }];


  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs
  );

  let results = (outs
    AnyTensor:$result
  );

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
}



def VN_ReduceOp:VN_Op<"reduce">{

  let summary = "Virtual Neuron reduction - reduces tensor to scalar";

  let arguments = (ins
    AnyTensor:$input
  );

  let results = (outs
    AnyFloat:$result
  );

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";

}

def VN_MergeOp : VN_Op<"merge"> {
  let summary = "Merge multiple scalar values into a tensor";
  
  let arguments = (ins
    Variadic<AnyFloat>:$inputs
  );
  
  let results = (outs
    AnyTensor:$result
  );
  
  let assemblyFormat = "$inputs attr-dict `:` functional-type(operands, $result)";
}

