namespace TensorFlowSharp.Tests

open TensorFlow
open Xunit

module NeuralNetOperationTests = 

    [<Theory>]
    [<InlineData(0.1f)>]
    [<InlineData(0.5f)>]
    [<InlineData(1.0f)>]
    let Should_ApplyDropout_ForFloatDataType(keep_prob:float32) =
        use graph = new TFGraph()
        use session = new TFSession(graph)

        let inputs = Array2D.init 4000 200 (fun i j -> float32(i+j+1))

        let a = graph.Placeholder(TFDataType.Float, new TFShape(4000L, 200L)) // create symbolic variable x
        let b = graph.Placeholder(TFDataType.Float, new TFShape(4000L, 200L)) // create symbolic variable keep_prob

        let y = graph.Dropout(a, b) // apply dropout to symbolic variables
        
        // evaluate expression with parameters for a and b
        let res = 
            session.Run([| a; b |], 
                        [| new TFTensor(inputs); new TFTensor(keep_prob) |], 
                        [| y |])
        
        let resTensor = res.[0]
        let resValue = resTensor.GetValue() :?> float32[,]

        let countZeros arr = arr |> Seq.cast<float32> |> Seq.filter (fun x -> x = 0.0f) |> Seq.length
        let numberOfOnes = countZeros resValue |> float32;
        let totalLength = resValue |> Seq.cast<float32> |> Seq.length |> float32
        let actualRatio = numberOfOnes / totalLength;
        let expected = 1.0f - keep_prob : float32

        Assert.True(System.Math.Abs(expected - actualRatio) < 0.05f)
