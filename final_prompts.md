# Final Prompts Used for LLM-Based Symptom Annotation

This file contains the final German zero-shot prompts used for automated annotation of four clinical symptoms in emergency department reports.  
For each report, the respective prompt was combined with the extracted clinical text and submitted to the locally deployed LLM.  
The model was instructed to return only one binary answer: `#ja#` or `#nein#`.

## Nausea

```text
Leidet der Patient an Übelkeit? Dann antworte mit #ja#. Wenn keine Übelkeit vorliegt oder keine Information dazu vorhanden ist, antworte mit #nein#. Erbrechen deutet auf Übelkeit hin. Gib keine Begründung aus.
```

## Vomiting

```text
Leidet der Patient an Erbrechen? Gehe exakt vor bei der Beantwortung. Antworte mit #ja#, wenn Erbrechen vorliegt. Wenn kein Erbrechen vorliegt oder keine Information dazu vorhanden ist, antworte mit #nein#. Wenn ein Patient Übelkeit hat, aber Erbrechen nicht explizit benannt wird, antworte mit #nein#. Gib keine Begründung aus.
```

## Diarrhea

```text
Leidet der Patient an Durchfall? Gehe exakt vor bei der Beantwortung. Antworte mit #ja#, wenn Durchfall vorliegt. Wenn kein Durchfall vorliegt oder keine Information dazu vorhanden ist, antworte mit #nein#. Gib keine Begründung aus.
```

## Dysuria

```text
Leidet der Patient an Dysurie? Gehe exakt vor bei der Beantwortung. Antworte mit #ja#, wenn Dysurie vorliegt. Wenn keine Dysurie vorliegt oder keine Information dazu vorhanden ist, antworte mit #nein#. Gib keine Begründung aus.
```
