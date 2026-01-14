func.func @matmul_example_lowered(
  %A : tensor<2x3xf32>,
  %B : tensor<3x2xf32>
) -> tensor<2x2xf32> {

  // C[0,0]
  %a_row_0 = vn.extract_tensor_row %A, 0
               : tensor<2x3xf32> -> !vn.vn1d<3xf32>
  %b_col_0 = vn.extract_tensor_col %B, 0
               : tensor<3x2xf32> -> !vn.vn1d<3xf32>

  %mul_00 = vn.mul %a_row_0, %b_col_0
               : (!vn.vn1d<3xf32>, !vn.vn1d<3xf32>) -> !vn.vn1d<3xf32>

  %c00 = vn.reduce %mul_00  //rotate and reduce 
           : !vn.vn1d<3xf32> -> !vn.vn1d<3xf32> // [ct, ct ,ct]
                                     //             ^
                                    //              |
                                   //  actuall c00 ( but encrypted)



  // C[0,1] 
  %a_row_0_b = vn.extract_tensor_row %A, 0
                 : tensor<2x3xf32> -> !vn.vn1d<3xf32>
  %b_col_1 = vn.extract_tensor_col %B, 1
               : tensor<3x2xf32> -> !vn.vn1d<3xf32>

  %mul_01 = vn.mul %a_row_0_b, %b_col_1
               : (!vn.vn1d<3xf32>, !vn.vn1d<3xf32>) -> !vn.vn1d<3xf32>

  %c01 = vn.reduce %mul_01
           : !vn.vn1d<3xf32> -> !vn.vn1d<3xf32>


  // C[1,0]
  %a_row_1 = vn.extract_tensor_row %A, 1
               : tensor<2x3xf32> -> !vn.vn1d<3xf32>
  %b_col_0_b = vn.extract_tensor_col %B, 0
                 : tensor<3x2xf32> -> !vn.vn1d<3xf32>

  %mul_10 = vn.mul %a_row_1, %b_col_0_b
               : (!vn.vn1d<3xf32>, !vn.vn1d<3xf32>) -> !vn.vn1d<3xf32>

  %c10 = vn.reduce %mul_10
           : !vn.vn1d<3xf32> -> !vn.vn1d<3xf32>


  // C[1,1]
  %a_row_1_b = vn.extract_tensor_row %A, 1
                 : tensor<2x3xf32> -> !vn.vn1d<3xf32>
  %b_col_1_b = vn.extract_tensor_col %B, 1
                 : tensor<3x2xf32> -> !vn.vn1d<3xf32>

  %mul_11 = vn.mul %a_row_1_b, %b_col_1_b
               : (!vn.vn1d<3xf32>, !vn.vn1d<3xf32>) -> !vn.vn1d<3xf32>

  %c11 = vn.reduce %mul_11
           : !vn.vn1d<3xf32> -> !vn.vn1d<3xf32>


  %C = vn.merge %c00, %c01, %c10, %c11 {reshape = (2,2)}
         : (!vn.vn1d<3xf32>, !vn.vn1d<3xf32>,!vn.vn1d<3xf32>,!vn.vn1d<3xf32>) -> tensor<2x2xf32>

  return %C : tensor<2x2xf32>
}
