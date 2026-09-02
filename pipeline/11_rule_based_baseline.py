import re

import pandas as pd

from config import (
    BASELINE_OUTPUT_DIR,
    SPLIT_DIR,
    TEXT_FILE,
)


SYMPTOM_PATTERNS = {
    "nausea": [
        r"\b(?:übelkeit|uebelkeit)\w*\b",
        r"\bnausea\b",
        r"\bnauseös\w*\b",
        r"\bnauseoes\w*\b",
        r"\b(?:übel|uebel)\b",
        r"\bbrechreiz\b",
    ],
    "vomiting": [
        r"\b(?:blut|kaffeesatz)?erbrechen(?:s)?\b",
        r"\berbrochen\b",
        r"\berbricht\b",
        r"\berbrach(?:en)?\b",
        r"\berbreche(?:n)?\b",
        r"\bemesis\b",
        r"\b(?:hämatemesis|haematemesis)\b",
        r"\bvomitus\b",
        r"\bsich(?:\s+\w+){0,3}\s+übergeben\b",
        r"\b(?:übergibt|übergab|übergebe)\s+sich\b",
        r"\b(?:kotzen|kotzte|gekotzt)\b",
    ],
    "diarrhea": [
        r"\bdurchf(?:all|älle(?:n)?|aelle(?:n)?)\b",
        r"\bdiarrh(?:ö(?:e|en)?|oe(?:n)?)\b",
        r"\bdiarrh(?:ö|oe)isch\w*\b",
    ],
    "dysuria": [
        r"\bdysuri(?:e|sch\w*)\b",
        r"\balguri(?:e|sch\w*)\b",
        r"\bmiktionsschmerz\w*\b",
        (
            r"\b(?:schmerz\w*|brennen)\b"
            r"(?:\s+\w+){0,4}\s+"
            r"\b(?:beim|bei|bei der|während der|während des)\s+"
            r"(?:miktion\w*|wasserlass\w*|urinier\w*)\b"
        ),
        (
            r"\b(?:schmerzhaft\w*|brennend\w*)\b"
            r"(?:\s+\w+){0,3}\s+"
            r"\b(?:miktion\w*|wasserlass\w*|urinier\w*)\b"
        ),
        (
            r"\b(?:miktion\w*|wasserlass\w*|urinier\w*)\b"
            r"(?:\s+\w+){0,4}\s+"
            r"\b(?:schmerz\w*|brennen|brennend\w*)\b"
        ),
    ],
}

NEGATION_PATTERNS = [
    r"\bkein(?:e|en|er|em|es)?\b",
    r"\bkeinerlei\b",
    r"\bohne\b",
    r"\bnicht\b",
    r"\bweder\b",
    r"\bvernein\w*\b",
    r"\bnegier\w*\b",
    r"\bfrei\s+von\b",
    r"\bnicht\s+vorhanden\b",
    r"\b(?:liegt|lag)\s+nicht\s+vor\b",
    r"\bnein\b",
]

WHO_CONSISTENCY = (
    r"\b(?:"
    r"wässrig\w*|waessrig\w*|"
    r"flüssig\w*|fluessig\w*|"
    r"dünnflüssig\w*|duennfluessig\w*|"
    r"ungeformt\w*"
    r")\b"
)

WHO_STOOL = (
    r"\b(?:"
    r"stuhl\w*|"
    r"stuhlgäng\w*|stuhlgaeng\w*"
    r")\b"
)

WHO_FREQUENCY = (
    r"\b(?:"
    r"(?:[3-9]|\d{2,})\s*(?:[-–]\s*\d{1,2})?\s*(?:x|×|mal)|"
    r"(?:drei|vier|fünf|fuenf|sechs|sieben|acht|neun|zehn)\s*mal|"
    r"(?:[3-9]|\d{2,})\s+(?:\w+\s+){0,2}"
    r"(?:stuhl\w*|stuhlgäng\w*|stuhlgaeng\w*)"
    r")\b"
)

WHO_TIME = (
    r"\b(?:"
    r"heute|seit\s+heute|seit\s+gestern|"
    r"täglich|taeglich|"
    r"(?:pro|am)\s+tag|"
    r"(?:innerhalb\s+)?24\s*(?:h|stunden)|"
    r"/\s*24\s*h"
    r")\b"
)


def normalize_text(text):
    return re.sub(
        r"\s+",
        " ",
        str(text).lower(),
    ).strip()


def split_sentences(text):
    return [
        sentence.strip()
        for sentence in re.split(
            r"[.!?;]+",
            normalize_text(text),
        )
        if sentence.strip()
    ]


def local_context(sentence, start, end, window=5):
    before = sentence[:start].split()[-window:]
    mention = sentence[start:end]
    after = sentence[end:].split()[:window]

    return " ".join(
        before + [mention] + after
    )


def is_negated(sentence, match):
    context = local_context(
        sentence,
        match.start(),
        match.end(),
    )

    return any(
        re.search(pattern, context)
        for pattern in NEGATION_PATTERNS
    )


def direct_match(sentence, patterns):
    for pattern in patterns:
        for match in re.finditer(pattern, sentence):
            if not is_negated(sentence, match):
                return True

    return False


def who_diarrhea_match(sentence):
    consistency_matches = list(
        re.finditer(
            WHO_CONSISTENCY,
            sentence,
        )
    )

    stool_matches = list(
        re.finditer(
            WHO_STOOL,
            sentence,
        )
    )

    if not consistency_matches or not stool_matches:
        return False

    if not re.search(WHO_FREQUENCY, sentence):
        return False

    if not re.search(WHO_TIME, sentence):
        return False

    consistency_positive = any(
        not is_negated(sentence, match)
        for match in consistency_matches
    )

    stool_positive = any(
        not is_negated(sentence, match)
        for match in stool_matches
    )

    return consistency_positive and stool_positive


def classify(text, symptom):
    for sentence in split_sentences(text):
        if direct_match(
            sentence,
            SYMPTOM_PATTERNS[symptom],
        ):
            return "ja"

        if (
            symptom == "diarrhea"
            and who_diarrhea_match(sentence)
        ):
            return "ja"

    return "nein"


def main():
    BASELINE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    texts = pd.read_csv(
        TEXT_FILE,
        dtype={"id": str},
    )

    for symptom in SYMPTOM_PATTERNS:
        split = pd.read_csv(
            SPLIT_DIR / f"split_{symptom}.csv",
            dtype={"id": str},
        )

        data = texts.merge(
            split,
            on="id",
            validate="one_to_one",
        )

        data = data[
            data["split"] == "validation"
        ].copy()

        data = data.sort_values(
            "id"
        ).reset_index(
            drop=True
        )

        data["prediction"] = data["text"].apply(
            lambda text: classify(
                text,
                symptom,
            )
        )

        output = data[
            [
                "id",
                "split",
                "prediction",
            ]
        ].copy()

        output.insert(
            1,
            "symptom",
            symptom,
        )

        output.to_csv(
            BASELINE_OUTPUT_DIR
            / f"baseline_{symptom}_validation.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"{symptom}: "
            f"{len(output)} predictions saved"
        )


if __name__ == "__main__":
    main()
