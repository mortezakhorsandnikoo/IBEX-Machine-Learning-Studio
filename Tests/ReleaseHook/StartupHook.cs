using System.Reflection;

// Verification-only startup hook: runs the smoke harness inside the actual published EXE.
// This library is not part of the shipped application.
internal static class StartupHook
{
    public static void Initialize()
    {
        string harness = Environment.GetEnvironmentVariable("IBEX_RELEASE_HARNESS");
        if (string.IsNullOrEmpty(harness)) return;
        int result = 1;
        var thread = new Thread(() =>
        {
            string directory = Environment.GetEnvironmentVariable("IBEX_RELEASE_OUTPUT");
            Directory.CreateDirectory(directory);
            using var log = new StreamWriter(Path.Combine(directory, "checks.log")) { AutoFlush = true };
            Console.SetOut(log); Console.SetError(log);
            try
            {
                var assembly = Assembly.LoadFrom(harness);
                result = (int)assembly.GetType("Smoke", true).GetMethod("Main", BindingFlags.Public | BindingFlags.Static).Invoke(null, new object[] { new[] { directory } });
            }
            catch (Exception ex) { log.WriteLine(ex); }
        });
        thread.SetApartmentState(ApartmentState.STA);
        thread.Start(); thread.Join();
        Environment.Exit(result);
    }
}
