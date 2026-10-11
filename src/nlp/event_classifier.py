from typing import Dict

EVENT_KEYWORDS = {
    "Geopolitical": ["war","conflict","sanction","tension","geopolit","military",
                     "invasion","ceasefire","strait","nato","missile","naval"],
    "Macroeconomic": ["fed","inflation","cpi","rate cut","rate hike","gdp",
                      "recession","jobs report","unemployment","central bank",
                      "stimulus","tariff","opec","treasury","pmi","yield"],
    "Credit Event": ["default","downgrade","bankruptcy","insolven","restructur",
                     "credit rating","missed payment","chapter 11","moody",
                     "s&p rating","fitch","junk"],
    "Merger/Acquisition": ["acqui","merger","takeover","buyout","stake",
                           "spin-off","spinoff","deal","combine"],
    "Product Launch": ["launch","unveil","release","new product","approval",
                       "fda","patent","rollout","debut","headset","chip"],
    "Regulatory": ["antitrust","regulat","fine","sec ","investigat","probe",
                   "lawsuit","ruling","court","doj","eu "],
    "Earnings": ["earnings","q1","q2","q3","q4","revenue","profit","beats",
                 "misses","guidance","outlook","dividend","buyback"],
    "Operational": ["recall","outage","breach","hack","cyber","strike",
                    "delay","disruption","ceo steps down","job cuts"],
}

PRIORS = {
    "Geopolitical": 0.8, "Credit Event": 0.9, "Macroeconomic": 0.7,
    "Merger/Acquisition": 0.6, "Regulatory": 0.55, "Earnings": 0.4,
    "Product Launch": 0.3, "Operational": 0.5, "Other": 0.3,
}


class EventClassifier:
    def __init__(self, use_zero_shot: bool = False,
                 model_name: str = "facebook/bart-large-mnli"):
        self.use_zero_shot = use_zero_shot
        self.pipe = None
        self.labels = list(EVENT_KEYWORDS.keys())
        if use_zero_shot:
            try:
                from transformers import pipeline
                self.pipe = pipeline("zero-shot-classification", model=model_name)
            except Exception as e:
                print(f"Zero-shot unavailable: {e}; using keyword fallback.")

    def classify(self, text: str) -> Dict:
        if self.pipe is not None:
            try:
                out = self.pipe(text[:512], candidate_labels=self.labels)
                return {"event": out["labels"][0],
                        "event_confidence": float(out["scores"][0]),
                        "method": "zero-shot"}
            except Exception as e:
                print(f"Zero-shot failed: {e}")
        t = text.lower()
        scores = {}
        for label, kws in EVENT_KEYWORDS.items():
            hits = sum(1 for kw in kws if kw in t)
            if hits:
                scores[label] = hits
        if not scores:
            return {"event": "Other", "event_confidence": 0.3, "method": "keyword"}
        best = max(scores.items(), key=lambda x: x[1])
        total = sum(scores.values())
        return {"event": best[0], "event_confidence": round(best[1] / total, 3),
                "method": "keyword"}


def event_prior(event: str) -> float:
    return PRIORS.get(event, 0.3)
