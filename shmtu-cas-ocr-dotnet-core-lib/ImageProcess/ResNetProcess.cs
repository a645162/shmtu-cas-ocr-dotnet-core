using System.Drawing;
using System.Runtime.Versioning;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace shmtu.core.cas.ocr.ImageProcess;

public static class ResNetProcess
{
    private static readonly float[] MeanValues = [123.675f, 116.28f, 103.53f];
    private static readonly float[] NormValues = [1.0f / 58.395f, 1.0f / 57.12f, 1.0f / 57.375f];

    [SupportedOSPlatform("windows6.2")]
    [SupportedOSPlatform("macos")]
    [SupportedOSPlatform("linux")]
    private static Bitmap LoadImage(string path)
    {
        return new Bitmap(path);
    }

    [SupportedOSPlatform("windows6.2")]
    [SupportedOSPlatform("macos")]
    [SupportedOSPlatform("linux")]
    public static DenseTensor<float> ConvertImageToTensor(Bitmap image)
    {
        const int width = 224;
        const int height = 224;
        const int channels = 3; // RGB

        var resizedImage = new Bitmap(image, new Size(width, height));
        var tensor = new DenseTensor<float>(new[] { 1, channels, height, width });

        for (var y = 0; y < height; y++)
        for (var x = 0; x < width; x++)
        {
            var pixel = resizedImage.GetPixel(x, y);

            tensor[0, 0, y, x] = (pixel.R - MeanValues[0]) * NormValues[0];
            tensor[0, 1, y, x] = (pixel.G - MeanValues[1]) * NormValues[1];
            tensor[0, 2, y, x] = (pixel.B - MeanValues[2]) * NormValues[2];
        }

        return tensor;
    }
}