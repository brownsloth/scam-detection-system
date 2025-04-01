// app/page.tsx
"use client";

import { useState } from "react";
import Explanation from "@/components/Explanation";

type ExplanationResult = {
  input: string;
  explanation: [string, number][];
};

export default function Home() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<ExplanationResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleExplain = async () => {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(`http://localhost:8000/explain?text=${encodeURIComponent(text)}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error("Explanation failed", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="max-w-2xl mx-auto py-12 px-4">
      <h1 className="text-4xl font-bold text-center mb-10 text-blue-300">🧠 Fake News Explainer</h1>

      <div className="space-y-4">
        <textarea
          className="w-full p-4 rounded bg-gray-900 text-white border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
          rows={4}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter a political claim, news headline, or statement..."
        />

        <div className="text-center">
          <button
            onClick={handleExplain}
            disabled={loading}
            className="px-6 py-2 bg-blue-600 text-white font-semibold rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? "Analyzing..." : "Explain"}
          </button>
        </div>
      </div>

      {result && (
        <div className="mt-8">
          <Explanation input={result.input} explanation={result.explanation} />
        </div>
      )}
    </main>
  );
}
