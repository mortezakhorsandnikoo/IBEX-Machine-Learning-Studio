using System;
using System.IO;
using System.Security.Cryptography;
using System.Windows;

namespace ArchiAI.Help
{
    public static class HelpDownloads
    {
        public const string CheatSheetFileName = "IBEX_Machine_Learning_Cheat_Sheet.pdf";
        public static byte[] ReadCheatSheet()
        {
            using var source = Application.GetResourceStream(HelpCatalog.ResourceUri(CheatSheetFileName)).Stream;
            using var buffer = new MemoryStream();
            source.CopyTo(buffer);
            return buffer.ToArray();
        }
        public static void SaveCheatSheet(string path) => File.WriteAllBytes(path, ReadCheatSheet());
        public static void OpenCheatSheet(Window owner)
        {
            try
            {
                var path = GetCheatSheetForViewing();
                System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo(path) { UseShellExecute = true });
            }
            catch (Exception ex)
            {
                MessageBox.Show(owner, "The PDF could not be opened. Open Help and user guide and choose Save cheat sheet PDF. Then open the saved file in a PDF reader and press Ctrl+P to print.\n\n" + ex.Message,
                    "Open cheat sheet", MessageBoxButton.OK, MessageBoxImage.Information);
            }
        }
        public static string GetCheatSheetForViewing()
        {
            var bytes = ReadCheatSheet();
            // Content-versioned paths let a new release coexist with an open older PDF.
            string version = Convert.ToHexString(SHA256.HashData(bytes)).Substring(0, 16);
            string folder = Path.Combine(Path.GetTempPath(), "IBEXMLStudio", "Help", version);
            Directory.CreateDirectory(folder);
            string path = Path.Combine(folder, CheatSheetFileName);
            if (!File.Exists(path) || !CryptographicOperations.FixedTimeEquals(SHA256.HashData(File.ReadAllBytes(path)), SHA256.HashData(bytes)))
                File.WriteAllBytes(path, bytes);
            return path;
        }
    }
}
