module {
  func.func @conv2d_1c1f(
    %input  : tensor<1x5x5xf32>,
    %kernel : tensor<1x3x3xf32>
  ) -> tensor<1x3x3xf32> {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index

   
    %w_slice = vn.extract_slice %kernel
      [%c0, %c0, %c0]
      [%c1, %c3, %c3]
      [1, 1, 1]
      : tensor<1x3x3xf32> -> !vn.vn1d<9xf32>


    // output (0,0)
    %p00 = vn.extract_slice %input
      [%c0, %c0, %c0]
      [%c1, %c3, %c3]
      [1, 1, 1]
      : tensor<1x5x5xf32> -> !vn.vn1d<9xf32>


    %mul00 = vn.mul %p00_flat, %w_flat
      : !vn.vn1d<9xf32>

    %mac00 = vn.reduce %mul00
      : !vn.vn1d<9xf32> -> f32


    // output (0,1) 
    %p01 = vn.extract_slice %input
      [%c0, %c0, %c1]
      [%c1, %c3, %c3]
      [1, 1, 1]
      : tensor<1x5x5xf32> -> !vn.vn1d<9xf32>


    %mul01 = vn.mul %p01_flat, %w_flat
      : !vn.vn1d<9xf32>

    %mac01 = vn.reduce %mul01
      : !vn.vn1d<9xf32> -> f32


    // output (0,2)
    %p02 = vn.extract_slice %input
      [%c0, %c0, arith.addi(%c1, %c1)]
      [%c1, %c3, %c3]
      [1, 1, 1]
      : tensor<1x5x5xf32> -> !vn.vn1d<9xf32>


    %mul02 = vn.mul %p02_flat, %w_flat
      : !vn.vn1d<9xf32>

    %mac02 = vn.reduce %mul02
      : !vn.vn1d<9xf32> -> f32


    // Merge all MAC results into output tensor
    %out = vn.merge
      [%mac00, %mac01, %mac02]
      : (f32, f32, f32) -> tensor<1x1x3xf32>

    return %out : tensor<1x1x3xf32>
  }
}



//for multi channel , multi filter 

// option 1 -> two reductions 
// for each output oc 
// for ic:
//   patch_ic  : vn1d<kh*kw>
//   weight_ic : vn1d<kh*kw>
//   dot → scalar
// sum over ic


//option 2  -> one reduction , but the size of  1d vector can be huge
// patch_all  : vn1d<(ic*kh*kw)>
// weight_all : vn1d<(ic*kh*kw)>
// dot(patch_all, weight_all) → scalar


//in both cases we need to stack up for each output oc 
//example : vn.merge of one output channel
// %out = vn.stack
//       [%mac00, %mac01, %mac02]
//       : (f32, f32, f32) -> tensor<1x1x3xf32>


