//
// Things to improve in the API
//   Need TFTensor constructors for the sake of F# as it does not use the implicit constructors
//   The C# Nullables are not surfaced in a way that makes it nice for F#, must manually call System.Nullable on it
//
open TensorFlow
open System.IO

let iconst (graph:TFGraph) (v:int) (label:string) =
    graph.Const (TFTensor.op_Implicit (v), label)

let fconst (graph:TFGraph) (v:float32) (label:string) =
        graph.Const (TFTensor.op_Implicit (v), label)

let constructGraphToNormalizeImage =
    use graph = new TFGraph ()
    let input = graph.Placeholder TFDataType.String

    let image = graph.Cast (graph.DecodeJpeg (contents = input, channels = System.Nullable 3L), DstT = TFDataType.Float)
    let expand = graph.ExpandDims (input = image, dim = graph.Const (TFTensor.op_Implicit (0), "make_batch"))
    let size = graph.Const (TFTensor.op_Implicit [|224;224 |], "size")
    let resize = graph.ResizeBilinear (images = expand, size = size)
    let mean = fconst graph 117.f "Mean"
    let scale = fconst graph 1.f "Scale"
    let output = graph.Div (x = graph.Sub (x = resize, y = mean), y = scale)
    (graph, input, output)

let createTensorFromImageFile f =
    let tensor = File.ReadAllBytes f |> TFTensor.CreateString
    let (graph, input, output) = constructGraphToNormalizeImage 
    use session = new TFSession (graph)
    let normalized = session.Run (runOptions = null, inputs = [| input |], inputValues = [| tensor |], outputs = [| output |])
    normalized.[0]


[<EntryPoint>]
let main argv = 
    use graph = new TFGraph()

    graph.Import (File.ReadAllBytes ("/tmp/tensorflow_inception_graph.pb"), "")
    use session = new TFSession (graph)

    0 // return an integer exit code

