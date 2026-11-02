using ArchiAI.ML;
using System.Collections.Generic;
using System.Linq;

namespace ArchiAI.Views
{
    public partial class MLView
    {
        private readonly List<PreprocessingStep> _preprocessingSteps = new();
        private PredictionRecipe _advPipeline;
        private PredictionRecipe _clustPipeline;
        private PredictionRecipe CapturePredictionRecipe(int target) => PredictionRecipe.Create(
            _dataset.Headers.Where((_, i) => i != target).ToArray(),
            target < 0 ? null : _dataset.Headers[target], _preprocessingSteps);
    }
}
