using System.Windows;
using System.Windows.Media;

namespace ArchiAI.Helpers
{
    public class DrawingVisualHost : FrameworkElement
    {
        private readonly DrawingVisual _v;
        public DrawingVisualHost(DrawingVisual v)
        { _v = v; AddVisualChild(v); AddLogicalChild(v); }
        protected override int VisualChildrenCount => 1;
        protected override Visual GetVisualChild(int i) => _v;
    }
}