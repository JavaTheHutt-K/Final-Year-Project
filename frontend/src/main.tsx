import React, { lazy, Suspense } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import App from "./App";
import "./styles.css";

const OverviewPage = lazy(() => import("./pages/OverviewPage"));
const DatasetPage = lazy(() => import("./pages/DatasetPage"));
const TrainingPage = lazy(() => import("./pages/TrainingPage"));
const ResultsPage = lazy(() => import("./pages/ResultsPage"));
const InsightsPage = lazy(() => import("./pages/InsightsPage"));
const AdaptationPage = lazy(() => import("./pages/AdaptationPage"));
const DevicesPage = lazy(() => import("./pages/DevicesPage"));
const SettingsPage = lazy(() => import("./pages/SettingsPage"));

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<App />}>
          <Route index element={<Suspense fallback={<div className="page-loading">Loading overview…</div>}><OverviewPage /></Suspense>} />
          <Route path="dataset" element={<Suspense fallback={<div className="page-loading">Loading traffic…</div>}><DatasetPage /></Suspense>} />
          <Route path="live-traffic" element={<Suspense fallback={<div className="page-loading">Loading traffic…</div>}><DatasetPage /></Suspense>} />
          <Route path="training" element={<Suspense fallback={<div className="page-loading">Loading clients…</div>}><TrainingPage /></Suspense>} />
          <Route path="clients" element={<Suspense fallback={<div className="page-loading">Loading clients…</div>}><TrainingPage /></Suspense>} />
          <Route path="results" element={<Suspense fallback={<div className="page-loading">Loading detections…</div>}><ResultsPage /></Suspense>} />
          <Route path="detections" element={<Suspense fallback={<div className="page-loading">Loading detections…</div>}><ResultsPage /></Suspense>} />
          <Route path="reports" element={<Suspense fallback={<div className="page-loading">Loading reports…</div>}><ResultsPage /></Suspense>} />
          <Route path="insights" element={<Suspense fallback={<div className="page-loading">Loading insights…</div>}><InsightsPage /></Suspense>} />
          <Route path="adaptation" element={<Suspense fallback={<div className="page-loading">Loading model performance…</div>}><AdaptationPage /></Suspense>} />
          <Route path="model-performance" element={<Suspense fallback={<div className="page-loading">Loading model performance…</div>}><AdaptationPage /></Suspense>} />
          <Route path="devices" element={<Suspense fallback={<div className="page-loading">Loading devices…</div>}><DevicesPage /></Suspense>} />
          <Route path="settings" element={<Suspense fallback={<div className="page-loading">Loading settings…</div>}><SettingsPage /></Suspense>} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
);
