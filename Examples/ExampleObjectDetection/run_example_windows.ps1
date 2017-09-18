function unzip($path,$to) {
    $7z = "$env:TEMP\7z"
    if (!(test-path $7z) -or !(test-path "$7z\7za.exe")) { 
        if (!(test-path $7z)) { md $7z | out-null }
        push-location $7z
        try {
            write-host "Downloading 7zip" -foregroundcolor cyan
            $wc = new-object system.net.webClient
            $wc.headers.add('user-agent', [Microsoft.PowerShell.Commands.PSUserAgent]::FireFox)
            $wc.downloadFile("http://www.7-zip.org/a/7za920.zip","$7z\7z.zip")
            write-host "done." foregroundcolor green

            add-type -assembly "system.io.compression.filesystem"
            [io.compression.zipfile]::extracttodirectory("$7z\7z.zip","$7z")
            del .\7z.zip
        }
        finally { pop-location }
    }

    if ($path.endswith('.tar.gz') -or $path.endswith('.tgz')) {
        # This is some crazy s**t right here
        $x = "cmd"
        $y = "/C `"^`"$7z\7za.exe^`" x ^`"$path^`" -so | ^`"$7z\7za.exe^`" x -y -si -ttar -o^`"$to^`""
        & $x $y
    } else {
        & "$7z\7za.exe" x $path -y -o"$to"
    }
}


$itemName = "faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017"
$fileName = $itemName + ".tar.gz"
$source = "http://download.tensorflow.org/models/object_detection/"+$fileName

$scriptPath = split-path -parent $MyInvocation.MyCommand.Definition
$destinationFolder =[io.path]::combine($scriptPath,$exeFolderRel)

$destination = $destinationFolder+$fileName
$destination

if(!(Test-Path -Path ([io.path]::combine($destinationFolder, $itemName) ))){
(new-object System.Net.WebClient).DownloadFile($source,$destination)
"model file downloaded"

unzip $destination $destinationFolder
"model is unzipped"
}
else{
"model folder already exists"
}


$imagesFolder = "test_images"
$imagesAbs = [io.path]::combine($scriptPath , $exeFolderRel , $imagesFolder)
Copy-Item ([io.path]::combine($scriptPath , $imagesFolder)) ([io.path]::combine($scriptPath ,$exeFolderRel)) -Recurse -force
("test image copied to path:" + $imagesAbs)

$mscocoFileName = "mscoco_label_map.pbtxt"
$mscoco = "https://raw.githubusercontent.com/tensorflow/models/master/object_detection/data/"+$mscocoFileName
(new-object System.Net.WebClient).DownloadFile($mscoco, [io.path]::combine($destinationFolder,$mscocoFileName))

"running object detection"
$run = [io.path]::combine($scriptPath , $exeFolderRel , "ExampleObjectDetection.exe")
$run

$inputImageArg = ("--input_image=" + [io.path]::combine($imagesAbs,"input.jpg"))
$outputImageArg =  ("--output_image="+[io.path]::combine($imagesAbs, "output.jpg"))
$catalogArg = ("--catalog=" + [io.path]::combine($scriptPath ,$exeFolderRel,"mscoco_label_map.pbtxt"))
$modelArg = ("--model="+[io.path]::combine($scriptPath , $exeFolderRel , $itemName ,"frozen_inference_graph.pb"))

"arguments"
$inputImageArg
$outputImageArg
$catalogArg
$modelArg

Start-Process -FilePath $run -ArgumentList $inputImageArg, $outputImageArg, $catalogArg, $modelArg -NoNewWindow -Wait

"detection completed"
