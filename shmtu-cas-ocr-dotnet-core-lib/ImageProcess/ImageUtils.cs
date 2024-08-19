using System.Drawing;
using System.Runtime.Versioning;

namespace shmtu.core.cas.ocr.ImageProcess;

public static class ImageUtils
{
    [SupportedOSPlatform("windows6.2")]
    [SupportedOSPlatform("macos")]
    [SupportedOSPlatform("linux")]
    public static Bitmap ConvertImageToBinary(Bitmap image, int threshold = 150)
    {
        // Create a temporary image with the same size as the original image
        var tempImage = new Bitmap(image.Width, image.Height);

        // Loop through each pixel in the image
        for (var x = 0; x < image.Width; x++)
        for (var y = 0; y < image.Height; y++)
        {
            // Get the color of the current pixel
            var currentColor = image.GetPixel(x, y);

            // Calculate the brightness value of the pixel (using a simple weighted average)
            var red = currentColor.R;
            var green = currentColor.G;
            var blue = currentColor.B;

            var luminance = (red * 299 + green * 587 + blue * 114) / 1000;

            // Determine if the pixel should be white or black based on the threshold
            // If the pixel is black, set it to black (usually 0, 0, 0)
            // If the pixel is white, set it to white (usually 255, 255, 255)
            tempImage.SetPixel(x, y, luminance >= threshold ? Color.White : Color.Black);
        }

        // Return the binary-processed image
        return tempImage;
    }

    [SupportedOSPlatform("windows6.2")]
    [SupportedOSPlatform("macos")]
    [SupportedOSPlatform("linux")]
    public static Bitmap RevertImageColor(Bitmap image)
    {
        // Create a temporary image with the same size as the original image
        var tempImage = new Bitmap(image.Width, image.Height);

        // Loop through each pixel in the image
        for (var x = 0; x < image.Width; x++)
        for (var y = 0; y < image.Height; y++)
        {
            // Get the color of the current pixel
            var currentColor = image.GetPixel(x, y);

            // Swap the red and blue channels and the green channel
            var red = currentColor.B;
            var green = currentColor.R;
            var blue = currentColor.G;

            // Set the new pixel color with swapped channels
            tempImage.SetPixel(x, y, Color.FromArgb(red, green, blue));
        }

        // Return the color-reverted image
        return tempImage;
    }
}