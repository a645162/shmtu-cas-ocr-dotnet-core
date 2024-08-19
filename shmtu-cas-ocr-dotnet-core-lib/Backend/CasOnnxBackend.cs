using System.Drawing;
using System.Runtime.Versioning;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using shmtu.core.cas.ocr.ImageProcess;
using shmtu.core.cas.ocr.Utils;

namespace shmtu.core.cas.ocr.Backend;

public class CasOnnxBackend
{
    private bool _isLoaded;

    private InferenceSession? _sessionDigit;
    private InferenceSession? _sessionEqualSymbol;
    private InferenceSession? _sessionOperator;

    public bool IsLoaded =>
        _isLoaded &&
        _sessionOperator != null &&
        _sessionDigit != null &&
        _sessionEqualSymbol != null;

    public static async Task<bool> DownloadModel(string directoryPath, IProgress<float>? progress = null)
    {
        const string baseUrl = ConstValue.ModelOnnxBaseUrl;

        string[] listFileName =
        {
            ConstValue.ModelOnnxEqualFp32,
            ConstValue.ModelOnnxOperatorFp32,
            ConstValue.ModelOnnxDigitFp32
        };

        try
        {
            using var client = new HttpClient();

            foreach (var fileName in listFileName)
            {
                var url = $"{baseUrl}/{fileName}";
                var localPath = Path.Combine(directoryPath, fileName);
                localPath = Path.GetFullPath(localPath) ?? ".";

                // Ensure the directory exists
                if (!Directory.Exists(localPath))
                    Directory.CreateDirectory(localPath);

                // Download the file with progress reporting
                await NetworkFile.DownloadFileAsync(client, url, localPath, progress);
            }

            return true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error downloading models: {ex.Message}");
            return false;
        }
    }

    public void LoadModel(string directoryPath)
    {
        _sessionEqualSymbol =
            new InferenceSession(
                Path.Combine(directoryPath, ConstValue.ModelOnnxEqualFp32)
            );
        _sessionOperator =
            new InferenceSession(
                Path.Combine(directoryPath, ConstValue.ModelOnnxOperatorFp32)
            );
        _sessionDigit =
            new InferenceSession(
                Path.Combine(directoryPath, ConstValue.ModelOnnxDigitFp32)
            );
        _isLoaded = true;
    }

    private static int PredictModel(InferenceSession? session, DenseTensor<float> inputTensor)
    {
        if (session == null) return -1;

        var inputs =
            new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

        // Run the model
        var results =
            session.Run(inputs);

        if (results.Count == 0)
            return -1;

        // Get the output tensor
        var outputTensor = results[0].AsTensor<float>();

        // Get the index of the highest probability
        var result = Array.IndexOf(outputTensor.ToArray(), outputTensor.Max());

        return result;
    }

    [SupportedOSPlatform("windows6.2")]
    [SupportedOSPlatform("macos")]
    [SupportedOSPlatform("linux")]
    private static int PredictResNet(InferenceSession? session, Bitmap image)
    {
        // Only for Debug
        // image.Save("BeforePredict.png");

        // Preprocess the image
        var inputTensor = ResNetProcess.ConvertImageToTensor(image);

        return PredictModel(session, inputTensor);
    }

    // Python Version:src/classify/predict/predict_file.py
    [SupportedOSPlatform("windows6.2")]
    [SupportedOSPlatform("macos")]
    [SupportedOSPlatform("linux")]
    public (int, string, int, int, int, int) PredictValidateCode(string imagePath)
    {
        return PredictValidateCode(new Bitmap(imagePath));
    }

    [SupportedOSPlatform("windows6.2")]
    [SupportedOSPlatform("macos")]
    [SupportedOSPlatform("linux")]
    public (int, string, int, int, int, int) PredictValidateCode(Bitmap originalImage)
    {
        var defaultValue = (-1, "", -1, -1, -1, -1);
        if (!IsLoaded) return defaultValue;

        if (originalImage.Width == 0 || originalImage.Height == 0) return defaultValue;

        var image = ImageUtils.ConvertImageToBinary(originalImage, CasCaptchaImage.ConfigThresh);
        image = ImageUtils.RevertImageColor(image);

        // Save Image for Debug
        // image.Save("input_1.png");

        var imageEqualSymbol =
            CasCaptchaImage.SplitImgByRatio(
                image,
                CasCaptchaImage.EqualSymbolKeyStart,
                CasCaptchaImage.EqualSymbolKeyEnd
            );
        var predictedEqualSymbol =
            (CasCaptchaImage.CasExprEqualSymbol)
            PredictResNet(_sessionEqualSymbol, imageEqualSymbol);

        var keyPoint =
            predictedEqualSymbol == CasCaptchaImage.CasExprEqualSymbol.Chs
                ? CasCaptchaImage.KeyPointChs
                : CasCaptchaImage.KeyPointSymbol;

        // Spilt Image
        var imageDigit1 =
            CasCaptchaImage.SplitImgByRatio(
                image,
                0,
                keyPoint[0]
            );
        var imageOperator =
            CasCaptchaImage.SplitImgByRatio(
                image,
                keyPoint[0],
                keyPoint[1]
            );
        var imageDigit2 =
            CasCaptchaImage.SplitImgByRatio(
                image,
                keyPoint[1],
                keyPoint[2]
            );

        // Predict
        var predictedOperator =
            (CasCaptchaImage.CasExprOperator)
            PredictResNet(_sessionOperator, imageOperator);

        var predictedDigit1 =
            PredictResNet(_sessionDigit, imageDigit1);
        var predictedDigit2 =
            PredictResNet(_sessionDigit, imageDigit2);

        // Calculate Result
        var result = CasCaptchaImage.CalculateOperator(
            predictedDigit1,
            predictedDigit2,
            predictedOperator
        );

        // Get Expr String
        var strOperator = CasCaptchaImage.GetOperatorString(predictedOperator);
        var expr = $"{predictedDigit1} {strOperator} {predictedDigit2} = {result}";

        return (
            result,
            expr,
            (int)predictedEqualSymbol,
            (int)predictedOperator,
            predictedDigit1,
            predictedDigit2
        );
    }

    public void Dispose()
    {
        _sessionEqualSymbol?.Dispose();
        _sessionOperator?.Dispose();
        _sessionDigit?.Dispose();

        _isLoaded = false;
    }

    ~CasOnnxBackend()
    {
        Dispose();
    }
}