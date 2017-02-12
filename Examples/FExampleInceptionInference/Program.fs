//
// Usage:
// mono FExampleInceptionInference file1.JPG file2.jpg...
//
// Things to improve in the F# API
//   The C# Nullables are not surfaced in a way that makes it nice for F#, must manually call System.Nullable on it
//
open TensorFlow
open System.IO
open System
open System.Net
open System.IO.Compression

//
// Downloads the inception graph and labels
//
let FetchModelFiles = 
    let inceptionUrl = "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"
    let dir = Path.Combine (Path.GetTempPath (), "Inception")
    let modelFile = Path.Combine (dir, "tensorflow_inception_graph.pb")
    let labelsFile = Path.Combine (dir, "imagenet_comp_graph_label_strings.txt")
    let zipfile = Path.Combine (dir, "inception5h.zip")

    if (not (File.Exists (modelFile) && File.Exists (labelsFile))) then
        printfn "Downloading inception5h model and labels"
        Directory.CreateDirectory (dir) |> ignore
        let wc = new WebClient ()
        wc.DownloadFile (inceptionUrl, zipfile)
        ZipFile.ExtractToDirectory (zipfile, dir)
        File.Delete (zipfile)

    (modelFile, labelsFile)

// Convenience functiosn to create tensor constants from an integer and a float
let iconst (graph:TFGraph) (v:int) (label:string) =
    graph.Const (new TFTensor (v), label)

let fconst (graph:TFGraph) (v:float32) (label:string) =
        graph.Const (new TFTensor (v), label)

// The inception model takes as input the image described by a Tensor in a very
// specific normalized format (a particular image size, shape of the input tensor,
// normalized pixel values etc.).
//
// This function constructs a graph of TensorFlow operations which takes as
// input a JPEG-encoded string and returns a tensor suitable as input to the
// inception model.
let constructGraphToNormalizeImage =
    // Some constants specific to the pre-trained model at:
    // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
    //
    // - The model was trained after with images scaled to 224x224 pixels.
    // - The colors, represented as R, G, B in 1-byte each were converted to
    //   float using (value - Mean)/Scale.
    let width = 224
    let height = 224
    let mean = 117.f
    let scale = 1.f

    let graph = new TFGraph ()
    let input = graph.Placeholder TFDataType.String
        
    let image = graph.Cast (graph.DecodeJpeg (contents = input, channels = System.Nullable 3L), DstT = TFDataType.Float)
    let expand = graph.ExpandDims (input = image, dim = graph.Const (TFTensor.op_Implicit (0), "make_batch"))
    let size = graph.Const (TFTensor.op_Implicit [|width;height |], "size")
    let resize = graph.ResizeBilinear (images = expand, size = size)
    let mean = fconst graph mean "Mean"
    let scale = fconst graph scale "Scale"
    let output = graph.Div (x = graph.Sub (x = resize, y = mean), y = scale)
    (graph, input, output)

// Convert the image in filename to a Tensor suitable as input to the Inception model.
let createTensorFromImageFile f =
    let tensor = File.ReadAllBytes f |> TFTensor.CreateString
    let (graph, input, output) = constructGraphToNormalizeImage 
    use session = new TFSession (graph)
    let normalized = session.Run (runOptions = null, inputs = [| input |], inputValues = [| tensor |], outputs = [| output |])
    normalized.[0]


[<EntryPoint>]
let main argv = 
    let (modelFile, labelFile) = FetchModelFiles
    let labels = File.ReadAllLines (labelFile)
    use graph = new TFGraph()
    graph.Import (File.ReadAllBytes (modelFile), "")
    use session = new TFSession (graph)
    let files = if argv.Length = 0 then [| "/tmp/demo.jpg" |] else argv

    for file in files do
        let tensor = createTensorFromImageFile file

        // Run the session
        let runner = session.GetRunner ();
        runner.AddInput (graph.["input"].[0], tensor)
        runner.Fetch (graph.["output"].[0]);
        let sessionOutput = runner.Run ();

        //
        // Other style of invoking the runner:
        // let sessionOutput = session.Run (inputs = [| graph.["input"].[0] |], inputValues= [| tensor |], outputs = [| graph.["output"].[0] |])   
        //
        let resultTensor = sessionOutput.[0]

        // The GetValue method returns an 'object' type, we need to cast this
        // to a float[] [] array, and the results will be on the first row.
        let probabilities = (resultTensor.GetValue (jagged = true) :?> float32 [] []).[0]
        let mutable bestIdx = 0
        let mutable best = 0.f
        for i in [0..probabilities.Length-1] do
            if probabilities.[i] > best then
                best <- probabilities.[i]
                bestIdx <- i
        printfn "%s - best match [%d]=%g %s" file bestIdx best labels.[bestIdx]
    0 // return an integer exit code

