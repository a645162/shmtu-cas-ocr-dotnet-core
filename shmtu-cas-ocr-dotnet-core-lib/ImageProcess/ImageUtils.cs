using System.Drawing;
using System.Runtime.Versioning;

namespace shmtu.core.cas.ocr.ImageProcess;

public class ImageUtils
{
    [SupportedOSPlatform("windows6.2")]
    public static Bitmap ConvertImageToBinary(Bitmap image, int threshold = 200)
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
            int red = currentColor.R;
            int green = currentColor.G;
            int blue = currentColor.B;
            var luminance = (red * 299 + green * 587 + blue * 114) / 1000;

            // Determine if the pixel should be white or black based on the threshold
            if (luminance >= threshold)
                // If the pixel is white, set it to white (usually 255, 255, 255)
                tempImage.SetPixel(x, y, Color.White);
            else
                // If the pixel is black, set it to black (usually 0, 0, 0)
                tempImage.SetPixel(x, y, Color.Black);
        }

        // Return the binary-processed image
        return tempImage;
    }
}