
include "VNTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"





class VN_Op<string mnemonic, list<Trait>traits = []>
    :Op<VN_Dialect,mnemonic,traits>{
        let cppNamespace = "::mlir::heir::vn";
    }


def VN_AddOp:VN_Op<"add", [Commutative, SameOperandsAndResultType]>{
    let summary = "Elementwise addition of all data within VN";
    let description =[{
        Adds two virtual neuron (1D) elementwise
    }];

    let arguments = (ins
    VN1D_Type : $lhs,
    VN1D_Type : $rhs
    );

    let results = (outs
        VN1D_Type:$result
    );

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:`  functional-type(operands, $result)";


}


def VN_MulOp : VN_Op<"mul", [Commutative,  SameOperandsAndResultType]> {
    let summary = "Elementwise multiplication of data within VN";

    let description = [{
        Multiply two virtual neuron (1D) elementwise
    }];


  let arguments = (ins
    VN1D_Type:$lhs,
    VN1D_Type:$rhs
  );

  let results = (outs
    VN1D_Type:$result
  );

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, $result)";
}



def VN_ReduceOp:VN_Op<"reduce">{

  let summary = "Virtual Neuron reduction";

  let arguments = (ins
    VN1D_Type:$input
  );

  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` qualified(type($result))";

}

def VN_MergeOp : VN_Op<"merge"> {
  let summary = "Merge two 1D tensors";

  let arguments = (ins
    VN1D_Type:$a,
    VN1D_Type:$b
  );

  let results = (outs
    VN1D_Type:$result
  );

  let assemblyFormat = "$a `,` $b attr-dict `:` functional-type(operands, $result)";
}

