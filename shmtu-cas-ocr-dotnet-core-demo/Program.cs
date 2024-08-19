// See https://aka.ms/new-console-template for more information

using shmtu.core.cas.ocr;

Console.WriteLine("Hello SHMTU CAS OCR .Net Core!");

Console.WriteLine($"OcrCoreLib Version:{OcrCoreLib.OcrCoreLibVersion()}");

var casOcr = new CasOcr
{
    // Set Location
    ModelDirectoryPath = "./Model"
};

Console.WriteLine($"ModelDirectoryPath:{casOcr.ModelDirectoryPath}");

var modelIsExist = casOcr.CheckModelIsExist();
Console.WriteLine($"ModelIsExist:{modelIsExist}");

if (!modelIsExist)
{
    // Download Model
    Console.WriteLine("Download Model");

    var progress = new Progress<float>(percent => { Console.Write($"\rDownload Progress: {percent:F2}%"); });

    var success = await casOcr.DownloadModel(progress);

    Console.WriteLine(success ? "\nAll files downloaded successfully!" : "\nFailed to download some files.");
}

// Load Model
Console.WriteLine("Load Model");

#pragma warning disable CA1416
casOcr.LoadModel();
#pragma warning restore CA1416

Console.WriteLine($"Model Is Loaded:{casOcr.IsLoaded}");

// Init Image Path
var basePath = "../../../../Example";
basePath = Path.GetFullPath(basePath) ?? ".";

string[] imageFileNameList =
[
    "test1_20240102160004_server.png",
    "test2_20240102160811_server.png",
    "test3_20240102160857_server.png",
    "test4_20240102160902_server.png",
    "test5_20240102160141_server.png",
    "test6_20240102160146_server.png"
];

// Predict
foreach (var filename in imageFileNameList)
{
    var path = Path.Combine(basePath, filename);
    Console.WriteLine($"{path}");

#pragma warning disable CA1416
    var result = casOcr.PredictValidateCode(path);
#pragma warning restore CA1416

    Console.WriteLine($"{result.Item2}");
}