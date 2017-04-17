﻿namespace TensorFlowSharp.Tests

open TensorFlow
open Xunit

module ArithmeticOperationTests = 

    [<Theory>]
    [<InlineData(1.0f, 2.0f, 2.0f)>]
    [<InlineData(3.0f, 3.0f, 9.0f)>]
    let Should_EvaluateMultiplyExpression_ForFloatDataType(aValue:float32 , bValue:float32, expected:float32) =
        use graph = new TFGraph()
        use session = new TFSession(graph)

        let a = graph.Placeholder(TFDataType.Float) // create symbolic variable a
        let b = graph.Placeholder(TFDataType.Float) // create symbolic variable b

        let y = graph.Mul(a, b) // multiply symbolic variables
        
        // evaluate expression with parameters for a and b
        let mul = 
            session.Run([| a; b |], 
                        [| new TFTensor(aValue); new TFTensor(bValue) |], 
                        [| y |])
        
        let mulTensor = mul.[0]
        let mulValue = mulTensor.GetValue() :?> float32

        Assert.Equal(expected, mulValue)
