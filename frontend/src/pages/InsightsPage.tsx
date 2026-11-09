import { useState } from "react";
import { generateAiInsights, getAIComparison, getApiErrorMessage } from "../api";
import { useDashboard } from "../DashboardContext";
import { EmptyState, PageHeading, Panel } from "../components";

function InsightsPage() {
  const { results, testSets, selectedTestSet, loading, running } = useDashboard();
  const [insights, setInsights] = useState("");
  const [comparison, setComparison] = useState<{
    testSet: string;
    analysis: string;
    source: "gemini" | "deterministic-fallback";
  } | null>(null);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [insightsError, setInsightsError] = useState<string | null>(null);
  const [comparisonError, setComparisonError] = useState<string | null>(null);
  const activeTestSet = selectedTestSet || testSets[0] || "";
  const comparisonMetrics = results?.metrics;
  const canCompare = Boolean(
    activeTestSet
      && comparisonMetrics?.centralised?.[activeTestSet]
      && comparisonMetrics.federated?.[activeTestSet],
  );

  const handleGenerateInsights = async () => {
    setInsightsLoading(true);
    setInsightsError(null);
    setInsights("");
    try {
      const response = await generateAiInsights();
      setInsights(
        `${response.source === "gemini" ? "Gemini-generated analysis" : "Deterministic fallback analysis"}\n\n${response.insights}`,
      );
    } catch (error) {
      setInsightsError(getApiErrorMessage(error));
    } finally {
      setInsightsLoading(false);
    }
  };

  const handleGenerateComparison = async () => {
    setComparisonLoading(true);
    setComparisonError(null);
    try {
      const response = await getAIComparison(activeTestSet);
      setComparison({
        testSet: response.test_set,
        analysis: response.analysis,
        source: response.source,
      });
    } catch (error) {
      setComparisonError(getApiErrorMessage(error));
    } finally {
      setComparisonLoading(false);
    }
  };

  return (
    <div className="insights-page">
      <PageHeading
        eyebrow="ANALYSIS"
        title="Insights"
        description="Explore generated findings from the latest experiment and get an explanation of the selected model comparison."
      />
      <Panel
        title="Experiment insights"
        subtitle="Generate a summary of the latest completed experiment."
        className="insights-panel"
      >
        <button
          className="button button-primary"
          onClick={handleGenerateInsights}
          disabled={insightsLoading || loading || running}
        >
          {insightsLoading ? "Generating insights…" : "Generate insights"}
        </button>
        {insightsError && <p className="ai-analysis-error" role="alert">{insightsError}</p>}
        {insights && <div className="ai-analysis-content insights-content" aria-live="polite">{insights}</div>}
        {!insights && !insightsError && (
          <p className="insights-hint">Insights will appear here after generation.</p>
        )}
      </Panel>
      <Panel
        title="Model comparison"
        subtitle={activeTestSet ? `Interpret the saved metrics for ${activeTestSet}.` : "Select an evaluated test set to compare models."}
        className="insights-panel"
      >
        <button
          className="button button-primary"
          onClick={handleGenerateComparison}
          disabled={!canCompare || running || comparisonLoading}
        >
          {comparisonLoading ? "Generating comparison…" : "Generate comparison"}
        </button>
        {comparisonError && <p className="ai-analysis-error" role="alert">{comparisonError}</p>}
        {comparison?.testSet === activeTestSet && (
          <div className="ai-analysis-content insights-content" aria-live="polite">
            <strong>
              {comparison.source === "gemini"
                ? "Gemini-generated analysis"
                : "Deterministic fallback analysis"}
            </strong>
            <br /><br />
            {comparison.analysis}
          </div>
        )}
        {!canCompare && !loading && (
          <div className="insights-empty">
            <EmptyState message="A completed centralised and federated evaluation is required for this comparison." />
          </div>
        )}
      </Panel>
    </div>
  );
}

export default InsightsPage;
