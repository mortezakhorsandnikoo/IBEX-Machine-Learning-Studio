namespace ArchiAI.Views
{
    public partial class MLView
    {
        private void ClearCrossValidationTables()
        {
            LinCvTable.Show(null); PolyCvTable.Show(null);
            AdvCvTable.Show(null); ClsCvTable.Show(null);
        }

        private CrossValidationTable ActiveCrossValidationTable() => MainTabs.SelectedIndex switch
        {
            2 => CmbRegAlgo.SelectedIndex == 0 ? LinCvTable : CmbRegAlgo.SelectedIndex == 1 ? PolyCvTable : AdvCvTable,
            3 => ClsCvTable,
            _ => null
        };
    }
}
