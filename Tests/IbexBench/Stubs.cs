// The application's PreprocessingEngine.cs contains "using static ArchiAI.Views.MLView;" and uses the
// nested LabelEncoderResult type, which lives in the WPF view. This stub provides only that type so the
// ML engines compile without WPF. (Copied from MLView.xaml.cs.)
using System.Collections.Generic;

namespace ArchiAI.Views
{
    public static class MLView
    {
        public class LabelEncoderResult
        {
            public string[] Headers { get; set; }
            public string[][] EncodedData { get; set; }
            public Dictionary<string, Dictionary<string, int>> Mappings { get; set; }
            public int[] EncodedColumnIndices { get; set; }
            public int TotalEncoded { get; set; }
        }
    }
}
