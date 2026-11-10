import { useState } from "react";
import { generateAiInsights, getAIComparison, getApiErrorMessage } from "../api";
import { useDashboard } from "../DashboardContext";
import { EmptyState, PageHeading, Panel } from "../components";

function InsightsPage() {
  const { results, testSets, selectedTestSet, loading, running } = useDashboard();
  const [question, setQuestion] = useState("");
  const [insights, setInsights] = useState<{
    text: string;
    source: "groq" | "deterministic-fallback";
  } | null>(null);
  const [comparison, setComparison] = useState<{
    testSet: string;
    analysis: string;
    source: "groq" | "deterministic-fallback";
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
    try {
      const response = await generateAiInsights(question);
      setInsights({ text: response.insights, source: response.source });
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
        title="Ask the experiment"
        subtitle="Ask a question about the latest completed experiment, or leave it blank for a general summary."
        className="insights-panel"
      >
        <div className="ai-input-card">
          <label htmlFor="insights-question">Your question</label>
          <textarea
            id="insights-question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="For example: Which model has the better recall, and what tradeoff does it show?"
            maxLength={2000}
            rows={4}
            disabled={insightsLoading || loading || running}
          />
          <div className="ai-input-footer">
            <span>{question.length}/2000 characters</span>
            <button
              className="button button-primary"
              onClick={handleGenerateInsights}
              disabled={insightsLoading || loading || running}
            >
              {insightsLoading ? "Generating insights…" : "Ask AI"}
            </button>
          </div>
        </div>
        {insightsError && <p className="ai-analysis-error" role="alert">{insightsError}</p>}
        {insights && (
          <div className="ai-output-card" aria-live="polite">
            <div className="ai-output-heading">
              <strong>AI insight</strong>
              <span>{insights.source === "groq" ? "Groq-generated" : "Deterministic fallback"}</span>
            </div>
            <div className="ai-analysis-content insights-content">{insights.text}</div>
          </div>
        )}
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
          <div className="ai-output-card" aria-live="polite">
            <div className="ai-output-heading">
              <strong>Model comparison insight</strong>
              <span>{comparison.source === "groq" ? "Groq-generated" : "Deterministic fallback"}</span>
            </div>
            <div className="ai-analysis-content insights-content">{comparison.analysis}</div>
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
