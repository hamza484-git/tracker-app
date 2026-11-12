import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { TrackingPage } from "./features/tracking/TrackingPage";
import { JobDetailPage } from "./features/tracking/JobDetailPage";
import { LiveFeed } from "./features/tracking/LiveFeed";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/tracking" replace />} />
        <Route path="/tracking" element={<TrackingPage />} />
        <Route path="/delivery-analysis" element={<JobDetailPage />} />
        <Route path="/live" element={<LiveFeed />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
);
