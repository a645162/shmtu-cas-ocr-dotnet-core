// See https://aka.ms/new-console-template for more information

using shmtu.core.cas.ocr;
using shmtu.core.cas.ocr.Backend;

Console.WriteLine("Hello SHMTU CAS OCR .Net Core!");

Console.WriteLine("OcrCoreLib Version:");
Console.WriteLine(OcrCoreLib.OcrCoreLibVersion());

var backend = new CasOnnxBackend();
backend.LoadModel(".");
Console.WriteLine(backend.IsLoaded);

string[] imagePathList =
[
    "./test1_20240102160004_server.png",
    "./test2_20240102160811_server.png",
    "./test3_20240102160857_server.png",
    "./test4_20240102160902_server.png",
    "./test5_20240102160141_server.png",
    "./test6_20240102160146_server.png"
];

foreach (var imagePath in imagePathList)
{
    Console.WriteLine($"{imagePath}");
    var result = backend.PredictValidateCode(imagePath);
    Console.WriteLine($"{result.Item2}");
}