using System.Drawing;
using System.Runtime.Versioning;
using shmtu.core.cas.ocr.Backend;

namespace shmtu.core.cas.ocr;

public class CasOcr
{
    private static string _modelDirectoryPath = ".";
    private readonly CasOnnxBackend _casOnnxBackend = new();

    public string ModelDirectoryPath
    {
        get => Path.GetFullPath(_modelDirectoryPath);
        set
        {
            // Check if the directory path is valid
            if (!Directory.Exists(value))
                // Create
                Directory.CreateDirectory(value);

            if (!Directory.Exists(value)) return;

            _modelDirectoryPath = Path.GetFullPath(value);
        }
    }

    public bool IsLoaded => _casOnnxBackend.IsLoaded;

    public async Task<bool> DownloadModel(IProgress<float>? progress)
    {
        return await CasOnnxBackend.DownloadModel(ModelDirectoryPath, progress);
    }

    public bool CheckModelIsExist()
    {
        return CasOnnxBackend.CheckModelIsExist(ModelDirectoryPath);
    }

    [SupportedOSPlatform("windows6.2")]
    [SupportedOSPlatform("macos")]
    [SupportedOSPlatform("linux")]
    public bool LoadModel()
    {
        if (_casOnnxBackend.IsLoaded) return true;

        try
        {
            _casOnnxBackend.LoadModel(ModelDirectoryPath);
            return true;
        }
        catch (Exception e)
        {
            Console.WriteLine(e);
            return false;
        }
    }

    [SupportedOSPlatform("windows6.2")]
    [SupportedOSPlatform("macos")]
    [SupportedOSPlatform("linux")]
    public (int, string, int, int, int, int) PredictValidateCode(string imagePath)
    {
        try
        {
            var image = new Bitmap(imagePath);
            return PredictValidateCode(image);
        }
        catch (Exception e)
        {
            Console.WriteLine(e);
            var defaultValue = (-1, "", -1, -1, -1, -1);
            return defaultValue;
        }
    }

    [SupportedOSPlatform("windows6.2")]
    [SupportedOSPlatform("macos")]
    [SupportedOSPlatform("linux")]
    public (int, string, int, int, int, int) PredictValidateCode(Bitmap image)
    {
        var defaultValue = (-1, "", -1, -1, -1, -1);
        if (!_casOnnxBackend.IsLoaded)
            if (!LoadModel())
                return defaultValue;

        if (!_casOnnxBackend.IsLoaded) return defaultValue;

        return _casOnnxBackend.PredictValidateCode(image);
    }

    public void Dispose()
    {
        _casOnnxBackend.Dispose();
    }

    ~CasOcr()
    {
        Dispose();
    }
}