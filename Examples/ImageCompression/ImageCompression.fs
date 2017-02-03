//
// Image compression using neural networks
//
// From models/compression/encoder.py
//
open System
open System.IO
open TensorFlow

let input = "example.png"
let iteration = 15
let output_codes = None
let model = "../../compression_residual_gru/residual_gru.pb"

let opt x = 
    System.Nullable x

// Convenience functiosn to create tensor constants from an integer and a float
let iconst (graph:TFGraph) (v:int) (label:string) =
    graph.Const (TFTensor.op_Implicit (v), label)

let input_tensor_names = 
    [| for a in 0 .. 16 do yield sprintf "loop_%02d/add:0" a |]

let output_tensor_names =
    let first = [|"GruBinarizer/SignBinarizer/Sign:0" |]
    Seq.append first [| for a in 0 .. 16 do yield sprintf "GruBinarizer/SignBinarizer/Sign_%d:0" a |] |> Seq.toArray

[<EntryPoint>]
let main argv = 
    use graph = new TFGraph()
    let outputs = [| for name in output_tensor_names do yield graph.[name] |];
    let input_image = graph.Placeholder TFDataType.String
    let input_image_str = File.ReadAllBytes (input) |> TFTensor.CreateString

    let decoded_image = if Path.GetExtension (input) = ".png" then graph.DecodePng (input_image, channels = opt 3L) else graph.DecodeJpeg (input_image, channels = opt 3L)
    let expanded_image = graph.ExpandDims (decoded_image, iconst graph 0 "zero")

    use session = new TFSession (graph)
    let result = session.Run (runOptions = null, inputs = [| input_image |], inputValues = [| input_image_str |], outputs = [| |], targetOpers = [| expanded_image.Operation |]);


    // The following will fail unless you rebuild your tensorflow to remove the 64mb limitation
    // https://github.com/tensorflow/tensorflow/issues/582
    graph.Import (new TFBuffer (File.ReadAllBytes (model)))
    let input_tensor = graph.["Placeholder:0"]

    0 // return an integer exit code

