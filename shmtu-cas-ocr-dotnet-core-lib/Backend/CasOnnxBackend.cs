using System.Drawing;
using System.Runtime.Versioning;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using shmtu.core.cas.ocr.ImageProcess;

namespace shmtu.core.cas.ocr.Backend;

public class CasOnnxBackend
{
    private InferenceSession? _digitSession;
    private InferenceSession? _equalSymbolSession;
    private bool _isLoaded;
    private InferenceSession? _operatorSession;

    public bool IsLoaded =>
        _isLoaded &&
        _operatorSession != null &&
        _digitSession != null &&
        _equalSymbolSession != null;

    public void LoadModel(string directoryPath)
    {
        _equalSymbolSession =
            new InferenceSession(
                Path.Combine(directoryPath, ConstValue.ModelOnnxEqualFp32)
            );
        _operatorSession =
            new InferenceSession(
                Path.Combine(directoryPath, ConstValue.ModelOnnxOperatorFp32)
            );
        _digitSession =
            new InferenceSession(
                Path.Combine(directoryPath, ConstValue.ModelOnnxDigitFp32)
            );
        _isLoaded = true;
    }

    private static int PredictModel(InferenceSession? session, DenseTensor<float> inputTensor)
    {
        if (session == null) return -1;

        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) };

        // Run the model
        var results =
            session.Run(inputs);

        if (results.Count == 0)
            return -1;

        // Get the output tensor
        var outputTensor = results.First().AsTensor<float>();

        // Get the index of the highest probability
        var result = Array.IndexOf(outputTensor.ToArray(), outputTensor.Max());

        return result;
    }

    [SupportedOSPlatform("windows6.2")]
    private static int PredictResNet(InferenceSession? session, Bitmap image)
    {
        // Preprocess the image
        var inputTensor = ResNetProcess.PreprocessImage(image);

        return PredictModel(session, inputTensor);
    }

    // Python Version:src/classify/predict/predict_file.py
    [SupportedOSPlatform("windows6.2")]
    public (int, string, int, int, int, int) PredictValidateCode(Bitmap image)
    {
        var defaultValue = (-1, "", -1, -1, -1, -1);
        if (!IsLoaded) return defaultValue;

        if (image.Width == 0 || image.Height == 0) return defaultValue;

        var binImage = ImageUtils.ConvertImageToBinary(image);

        var imageEqualSymbol =
            CasCaptchaImage.SplitImgByRatio(
                binImage
            );
        var predictedEqualSymbol =
            (CasCaptchaImage.CasExprEqualSymbol)
            PredictResNet(_equalSymbolSession, imageEqualSymbol);

        var keyPoint =
            predictedEqualSymbol == CasCaptchaImage.CasExprEqualSymbol.Chs
                ? CasCaptchaImage.KeyPointChs
                : CasCaptchaImage.KeyPointSymbol;

        // Spilt Image
        var imageDigit1 =
            CasCaptchaImage.SplitImgByRatio(
                binImage,
                0,
                keyPoint[0]
            );
        var imageOperator =
            CasCaptchaImage.SplitImgByRatio(
                binImage,
                keyPoint[0],
                keyPoint[1]
            );
        var imageDigit2 =
            CasCaptchaImage.SplitImgByRatio(
                binImage,
                keyPoint[1],
                keyPoint[2]
            );

        // Predict
        var predictedOperator =
            (CasCaptchaImage.CasExprOperator)
            PredictResNet(_operatorSession, imageOperator);

        var predictedDigit1 =
            PredictResNet(_digitSession, imageDigit1);
        var predictedDigit2 =
            PredictResNet(_digitSession, imageDigit2);

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
        _equalSymbolSession?.Dispose();
        _operatorSession?.Dispose();
        _digitSession?.Dispose();

        _isLoaded = false;
    }

    ~CasOnnxBackend()
    {
        Dispose();
    }
}