module {
  func.func @conv2d_flattened_vn(
    %input  : tensor<2x5x5xf32>,       
    %kernel : tensor<2x2x3x3xf32>      
  ) -> tensor<2x3x3xf32> {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index


    %w0 = vn.extract_slice %kernel
      [%c0, %c0, %c0, %c0]  
      [%c1, %c2, %c3, %c3] 
      [1, 1, 1, 1]          
      : tensor<2x2x3x3xf32> -> !vn.vn1d<18xf32>

    %w1 = vn.extract_slice %kernel
      [%c1, %c0, %c0, %c0]
      [%c1, %c2, %c3, %c3]
      [1, 1, 1, 1]
      : tensor<2x2x3x3xf32> -> !vn.vn1d<18xf32>


    // Output channel 0: MACs

    %r000 = vn.reduce (vn.mul (vn.extract_slice %input [%c0, %c0, %c0] [%c2, %c3, %c3] [1,1,1]), %w0) : !vn.vn1d<18xf32> -> f32
    %r001 = vn.reduce (vn.mul (vn.extract_slice %input [%c0, %c0, %c1] [%c2, %c3, %c3] [1,1,1]), %w0) : !vn.vn1d<18xf32> -> f32
    %r002 = vn.reduce (vn.mul (vn.extract_slice %input [%c0, %c0, %c2] [%c2, %c3, %c3] [1,1,1]), %w0) : !vn.vn1d<18xf32> -> f32

    %r010 = vn.reduce (vn.mul (vn.extract_slice %input [%c0, %c1, %c0] [%c2, %c3, %c3] [1,1,1]), %w0) : !vn.vn1d<18xf32> -> f32
    %r011 = vn.reduce (vn.mul (vn.extract_slice %input [%c0, %c1, %c1] [%c2, %c3, %c3] [1,1,1]), %w0) : !vn.vn1d<18xf32> -> f32
    %r012 = vn.reduce (vn.mul (vn.extract_slice %input [%c0, %c1, %c2] [%c2, %c3, %c3] [1,1,1]), %w0) : !vn.vn1d<18xf32> -> f32

    %r020 = vn.reduce (vn.mul (vn.extract_slice %input [%c0, %c2, %c0] [%c2, %c3, %c3] [1,1,1]), %w0) : !vn.vn1d<18xf32> -> f32
    %r021 = vn.reduce (vn.mul (vn.extract_slice %input [%c0, %c2, %c1] [%c2, %c3, %c3] [1,1,1]), %w0) : !vn.vn1d<18xf32> -> f32
    %r022 = vn.reduce (vn.mul (vn.extract_slice %input [%c0, %c2, %c2] [%c2, %c3, %c3] [1,1,1]), %w0) : !vn.vn1d<18xf32> -> f32

    // Merge scalars into output grid for OC0
    %out0 = vn.merge
      [%r000, %r001, %r002,
       %r010, %r011, %r012,
       %r020, %r021, %r022]
      shape = [1,3,3]
      : (f32 x 9) -> tensor<1x3x3xf32>

    // Output channel 1: MACs
    %r100 = vn.reduce (vn.mul (vn.extract_slice %input [%c0, %c0, %c0] [%c2, %c3, %c3] [1,1,1]), %w1) : !vn.vn1d<18xf32> -> f32
    %r101 = vn.reduce (vn.mul (vn.extract_slice %input [%c0, %c0, %c1] [%c2, %c3, %c3] [1,1,1]), %w1) : !vn.vn1d<18xf32> -> f32
    %r102 = vn.reduce (vn.mul (vn.extract_slice %input [%c0, %c0, %c2] [%c2, %c3, %c3] [1,1,1]), %w1) : !vn.vn1d<18xf32> -> f32

    %r110 = vn.reduce (vn.mul (vn.extract_slice %input [%c0, %c1, %c0] [%c2, %c3, %c3] [1,1,1]), %w1) : !vn.vn1d<18xf32> -> f32
    %r111 = vn.reduce (vn.mul (vn.extract_slice %input [%c0, %c1, %c1] [%c2, %c3, %c3] [1,1,1]), %w1) : !vn.vn1d<18xf32> -> f32
    %r112 = vn.reduce (vn.mul (vn.extract_slice %input [%c0, %c1, %c2] [%c2, %c3, %c3] [1,1,1]), %w1) : !vn.vn1d<18xf32> -> f32

    %r120 = vn.reduce (vn.mul (vn.extract_slice %input [%c0, %c2, %c0] [%c2, %c3, %c3] [1,1,1]), %w1) : !vn.vn1d<18xf32> -> f32
    %r121 = vn.reduce (vn.mul (vn.extract_slice %input [%c0, %c2, %c1] [%c2, %c3, %c3] [1,1,1]), %w1) : !vn.vn1d<18xf32> -> f32
    %r122 = vn.reduce (vn.mul (vn.extract_slice %input [%c0, %c2, %c2] [%c2, %c3, %c3] [1,1,1]), %w1) : !vn.vn1d<18xf32> -> f32

    %out1 = vn.merge
      [%r100, %r101, %r102,
       %r110, %r111, %r112,
       %r120, %r121, %r122]
      shape = [1,3,3]
      : (f32 x 9) -> tensor<1x3x3xf32>

    // Final output: stack channels
    %final = vn.stack
      [%out0, %out1]
      axis = 0
      : tensor<1x3x3xf32> -> tensor<2x3x3xf32>

    return %final : tensor<2x3x3xf32>
  }
}
