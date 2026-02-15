


include "mlir/IR/OpBase.td"
include "VNDialect.td"

class VNBaseType<string name , string typeMnemonic>
    :TypeDef<VN_Dialect,name>{
        let mnemonic = typeMnemonic;
    }


def VN1D_Type : VNBaseType<"VN1D","vn1d">{
    let summary = "A 1d dimensinal tensor type for virtual neuron";
    let description = [{
        The virtual neurons are expected to be a 1d vector
    }];

    let parameters = (ins
    "unsigned": $length,
    "Type":$elementType
    );

    let assemblyFormat = "`<`$length`x`$elementType`>`";

}


