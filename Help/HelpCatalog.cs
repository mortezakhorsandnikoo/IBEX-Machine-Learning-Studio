using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Windows;

namespace ArchiAI.Help
{
    public sealed class HelpBlock
    {
        public string Kind { get; set; }
        public string Text { get; set; }
        public string[] Items { get; set; }
        public string Image { get; set; }
        public string Caption { get; set; }
    }
    public sealed class HelpTopic
    {
        public string Id { get; set; }
        public string Category { get; set; }
        public string Title { get; set; }
        public string Summary { get; set; }
        public string Keywords { get; set; }
        public string[] Related { get; set; } = Array.Empty<string>();
        public HelpBlock[] Blocks { get; set; } = Array.Empty<HelpBlock>();
        public string SearchText => Title + " " + Summary + " " + Keywords + " " + string.Join(" ", Blocks.Select(b => b.Text + " " + b.Caption + " " + string.Join(" ", b.Items ?? Array.Empty<string>())));
    }
    public sealed class HelpMatch
    {
        public HelpTopic Topic { get; set; }
        public string Title => Topic.Title;
        public string Category => Topic.Category;
        public string Snippet { get; set; }
        public int Score { get; set; }
    }
    public static class HelpCatalog
    {
        public static Uri ResourceUri(string path) => new Uri("pack://application:,,,/IBEX.MLStudio;component/Help/" + path, UriKind.Absolute);
        public static IReadOnlyList<HelpTopic> Load()
        {
            using var stream = Application.GetResourceStream(ResourceUri("guide.json")).Stream;
            var topics = JsonSerializer.Deserialize<List<HelpTopic>>(stream);
            if (topics == null || topics.Count == 0 || topics.Select(t => t.Id).Distinct().Count() != topics.Count)
                throw new InvalidOperationException("The user guide could not be loaded.");
            return topics;
        }
        public static string[] Tokens(string query) => Regex.Matches((query ?? "").ToLowerInvariant(), @"[\p{L}\p{N}²]+").Cast<Match>().Select(m => m.Value).Distinct().Take(16).ToArray();
        private static string Normalize(string value) => string.Join(" ", Regex.Matches((value ?? "").ToLowerInvariant(), @"[\p{L}\p{N}²]+").Cast<Match>().Select(m => m.Value));
        public static IReadOnlyList<HelpMatch> Search(IEnumerable<HelpTopic> topics, string query, string category = null)
        {
            var tokens = Tokens(query);
            var matches = new List<HelpMatch>();
            foreach (var topic in topics)
            {
                if (category != null && topic.Category != category) continue;
                string text = topic.SearchText, normalized = Normalize(text);
                if (tokens.Any(t => !normalized.Contains(t, StringComparison.Ordinal))) continue;
                int score = tokens.Sum(t => Normalize(topic.Title).Contains(t, StringComparison.Ordinal) ? 20 : Normalize(topic.Keywords).Contains(t, StringComparison.Ordinal) ? 5 : 1);
                string snippet = topic.Summary;
                if (tokens.Length > 0)
                {
                    var block = topic.Blocks.FirstOrDefault(b => tokens.Any(t => Normalize(b.Text + " " + string.Join(" ", b.Items ?? Array.Empty<string>())).Contains(t, StringComparison.Ordinal)));
                    if (block != null)
                    {
                        string body = block.Text + " " + string.Join(" ", block.Items ?? Array.Empty<string>());
                        int first = tokens.Select(t => body.IndexOf(t, StringComparison.OrdinalIgnoreCase)).Where(i => i >= 0).DefaultIfEmpty(0).Min();
                        int start = Math.Max(0, first - 45);
                        snippet = (start > 0 ? "…" : "") + body.Substring(start, Math.Min(180, body.Length - start)).Trim() + (body.Length - start > 180 ? "…" : "");
                    }
                }
                matches.Add(new HelpMatch { Topic = topic, Snippet = snippet, Score = score });
            }
            return matches.OrderByDescending(m => m.Score).ToArray();
        }
    }
}
