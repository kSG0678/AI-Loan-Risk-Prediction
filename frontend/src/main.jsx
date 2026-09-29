import React from "react";
import { createRoot } from "react-dom/client";
import PredictionHistoryPage from "./pages/PredictionHistoryPage.jsx";
import PredictionPage from "./pages/PredictionPage.jsx";
import "./styles.css";

const Page =
  window.location.pathname.replace(/\/+$/, "") === "/history"
    ? PredictionHistoryPage
    : PredictionPage;

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Page />
  </React.StrictMode>,
);
