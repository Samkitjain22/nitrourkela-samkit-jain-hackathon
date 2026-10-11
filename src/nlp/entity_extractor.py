import re
from typing import Dict, List

DEFAULT_ALIASES = {
    "AAPL": ["apple", "aapl", "$aapl"],
    "MSFT": ["microsoft", "msft", "$msft"],
    "AMZN": ["amazon", "amzn", "$amzn"],
    "GOOGL": ["google", "alphabet", "googl", "$googl"],
    "META": ["meta", "facebook", "instagram", "$meta"],
    "NVDA": ["nvidia", "nvda", "$nvda"],
    "JPM": ["jpmorgan", "jp morgan", "jpm", "$jpm"],
    "V": ["visa", "$v"],
    "JNJ": ["johnson & johnson", "johnson and johnson", "jnj", "$jnj"],
    "WMT": ["walmart", "wmt", "$wmt"],
    "XOM": ["exxon", "exxon mobil", "xom", "$xom"],
    "PG": ["procter", "procter & gamble", "pg", "$pg"],
    "UNH": ["unitedhealth", "united health", "unh", "$unh"],
    "HD": ["home depot", "hd", "$hd"],
    "BAC": ["bank of america", "bac", "$bac"],
}


class EntityExtractor:
    def __init__(self, aliases: Dict[str, List[str]] | None = None):
        self.aliases = aliases or DEFAULT_ALIASES
        self.cashtag = re.compile(r"\$([A-Za-z]{1,5})\b")

    def extract(self, text: str) -> List[str]:
        found = set()
        low = text.lower()
        for ticker, names in self.aliases.items():
            for n in names:
                if n in low:
                    found.add(ticker)
                    break
        for m in self.cashtag.finditer(text):
            t = m.group(1).upper()
            if t in self.aliases:
                found.add(t)
        return sorted(found)
