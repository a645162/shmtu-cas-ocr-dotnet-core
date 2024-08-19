using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace shmtu.core.cas.ocr.ImageProcess;

public static class ResNetProcess
{
    private static readonly float[] MeanValues = [123.675f, 116.28f, 103.53f];
    private static readonly float[] NormValues = [1.0f / 58.395f, 1.0f / 57.12f, 1.0f / 57.375f];

    [SupportedOSPlatform("windows6.2")]
    private static Bitmap LoadImage(string path)
    {
        return new Bitmap(path);
    }

    [SupportedOSPlatform("windows6.2")]
    public static DenseTensor<float> PreprocessImage(Bitmap image)
    {
        // Define the dimensions of the input to the ResNet model
        const int width = 224;
        const int height = 224;
        const int channels = 3; // RGB

        // Resize the image to the required input dimensions
        var resizedImage = new Bitmap(image, new Size(width, height));

        // Lock the bitmap bits
        var bitmapData =
            resizedImage.LockBits(
                new Rectangle(0, 0, width, height),
                ImageLockMode.ReadOnly,
                PixelFormat.Format24bppRgb
            );

        // Define variables to store image's dimensions and pixel information
        var stride = bitmapData.Stride;
        var scan0 = bitmapData.Scan0;

        // Allocate a tensor to hold the image data
        var tensor = new DenseTensor<float>(new[] { 1, channels, height, width });

        // Create a byte array to hold the image data
        var bytesPerPixel = Image.GetPixelFormatSize(bitmapData.PixelFormat) / 8;
        var totalBytes = stride * height;
        var pixels = new byte[totalBytes];

        // Copy the image data to the byte array
        Marshal.Copy(scan0, pixels, 0, totalBytes);

        // Iterate over each pixel in the image and copy it to the tensor
        for (var y = 0; y < height; y++)
        for (var x = 0; x < width; x++)
        {
            // Compute index in the byte array
            var idx = y * stride + x * bytesPerPixel;

            // Subtract mean and divide by std deviation, then set the BGR values in the tensor
            tensor[0, 0, y, x] = (pixels[idx] / 255.0f - MeanValues[0]) * NormValues[0]; // B
            tensor[0, 1, y, x] = (pixels[idx + 1] / 255.0f - MeanValues[1]) * NormValues[1]; // G
            tensor[0, 2, y, x] = (pixels[idx + 2] / 255.0f - MeanValues[2]) * NormValues[2]; // R
        }

        // Unlock the bits of the image
        resizedImage.UnlockBits(bitmapData);

        return tensor;
    }
}