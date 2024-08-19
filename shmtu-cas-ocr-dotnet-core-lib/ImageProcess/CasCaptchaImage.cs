using System.Drawing;
using System.Runtime.Versioning;

namespace shmtu.core.cas.ocr.ImageProcess;

public static class CasCaptchaImage
{
    public enum CasExprEqualSymbol
    {
        Chs = 0,
        Symbol = 1
    }

    public enum CasExprOperator
    {
        Add = 0,
        AddChs = 1,
        Sub = 2,
        SubChs = 3,
        Mul = 4,
        MulChs = 5
    }

    // Python Version:src/config/config.py
    public const float EqualSymbolKeyStart = 0.7f;
    public const float EqualSymbolKeyEnd = 1.0f;
    public const int ConfigThresh = 200;
    public static readonly float[] KeyPointSymbol = [0.25f, 0.58f, 0.75f];
    public static readonly float[] KeyPointChs = [0.15f, 0.33f, 0.46f];

    public static string GetOperatorString(CasExprOperator exprOperator)
    {
        return exprOperator switch
        {
            CasExprOperator.Add or CasExprOperator.AddChs => "+",
            CasExprOperator.Sub or CasExprOperator.SubChs => "-",
            CasExprOperator.Mul or CasExprOperator.MulChs => "×",
            _ => ""
        };
    }

    public static int CalculateOperator(int digit1, int digit2, CasExprOperator exprOperator)
    {
        return exprOperator switch
        {
            CasExprOperator.Add or CasExprOperator.AddChs => digit1 + digit2,
            CasExprOperator.Sub or CasExprOperator.SubChs => digit1 - digit2,
            CasExprOperator.Mul or CasExprOperator.MulChs => digit1 * digit2,
            _ => -1
        };
    }

    [SupportedOSPlatform("windows6.2")]
    public static Bitmap SplitImgByRatio(Bitmap image, float startRatio = 0.7f, float endRatio = 1.0f)
    {
        var width = image.Width;
        var height = image.Height;

        var horizontalStart = (int)(width * startRatio);
        var horizontalEnd = (int)(width * endRatio);
        if (endRatio >= 1) horizontalEnd = width;

        var cropArea = new Rectangle(horizontalStart, 0, horizontalEnd - horizontalStart, height);
        var croppedImage = new Bitmap(cropArea.Width, cropArea.Height);

        using (var g = Graphics.FromImage(croppedImage))
        {
            g.DrawImage(image, new Rectangle(0, 0, croppedImage.Width, croppedImage.Height), cropArea,
                GraphicsUnit.Pixel);
        }

        return croppedImage;
    }
}