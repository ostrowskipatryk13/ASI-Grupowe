# Model Card

> Scaffold for Sprint 3 — do not implement before this sprint.
> Minimalną Model Card wypełnia zespół w Sprincie 3, a aktualizuje w Sprincie 6.

## Model

- Typ modelu i jego wersja (alias `production` w W&B): AutoGluon 'TabularPredictor', konfiguracja 'aq_fast'
- Data utworzenia i autor runu: 26.11.2026, Patryk Ostrowski

## Przeznaczenie

- Problem (binary classification) i klasa pozytywna: czy klient operatora telekomunikacyjnego
  odejdzie; target `Churn`, klasa pozytywna `1` (klient odszedł).
- Użytkownik systemu i decyzja, którą wspiera model: operator telekomunikacyjny. Model wskazuje klientów zagrożonych
  odejściem.
- Zastosowania, do których model nie jest przeznaczony:  automatyczne decyzje wobec klienta bez
  udziału człowieka (np. zmiana warunków umowy)

## Dane

- Dataset (id z `docs/datasets.md`) i fingerprint danych:`iranian_churn` (UCI Iranian Churn,
  CC BY 4.0, DOI https://doi.org/10.24432/C5JW3Z), 3150 rekordów, 15,71% klasy pozytywnej.
  SHA-256 pliku: `90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1`
- Strategia splitu z kontraktu i liczebności zbiorów: `stratified_random`, `test_size` 0,2,
  `random_state` 42 — 2520 rekordów treningowych i 630 testowych

## Metryki

- Metryka główna na zbiorze testowym, w porównaniu z DummyClassifier: Zbiór testowy (630 rekordów). DummyClassifier z `metrics.json` (Sprint 1), pozostałe modele
z pełnego `kedro run` z 26.11.2026, z którego pochodzi artefakt `v0`.
| Model | F1 | Precision | Recall |
|---|---|---|---|
| DummyClassifier | 0,000 | 0,000 | 0,000 |
| LogisticRegression (baseline) | 0,544 | 0,833 | 0,404 |
| **ag_fast (`production`)** | **0,893** | **0,860** | **0,929** |
 Metryka główna na zbiorze testowym, w porównaniu z DummyClassifier: F1 0,893 wobec 0,000
  (DummyClassifier zawsze przewiduje, że klient zostaje).
- Pozostałe metryki i macierz pomyłek: accuracy 0,965; tn 516, fp 15, fn 7, tp 92. Model
  wykrywa 92 z 99 odejść; 15 ze 107 alarmów jest fałszywych

## Ograniczenia i ryzyka

- Ograniczenia danych (z `docs/datasets.md`):
  Mały zbiór (3150 rekordów), więc metryki na
  zbiorze testowym mają dużą wariancję; dane jednego operatora z jednego okresu; część cech
  to kategorie zakodowane liczbami. Zgodnie z decyzją ze Sprintu 1 w danych zostało 300
  duplikatów: 95 z 630 rekordów testowych ma identyczną kopię w zbiorze treningowym, więc
  metryki mogą być zawyżone. `Age` to dana osobowa i dubluje `Age Group`.
- Znane błędy i sytuacje, w których model zawodzi:
  7 z 99 odchodzących klientów nie zostaje wykrytych (operator nie podejmie wobec nich działań), a 15 klientów, którzy zostają, dostaje
  fałszywy alarm (niepotrzebna oferta).
## Wersje

- Wersje bibliotek modelu i git SHA treningu: Python 3.11.18, autogluon.tabular / core /
  common 1.6.3, lightgbm 4.7.0, catboost 1.2.10, scikit-learn 1.7.2, pandas 2.3.3,
  numpy 2.2.6; git SHA `1333b7120a49e319b9f0bdc2154a6bcf1cfd9872`.
