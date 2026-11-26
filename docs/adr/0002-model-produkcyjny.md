# ADR-2: Model produkcyjny i jego postać jako artefaktu

- **Status:** proponowany
- **Sprint:** S3
- **Autor:** Patryk Ostrowski
- **Recenzent:**

## Kontekst

W Sprincie 3 porównaliśmy baseline i dwie konfiguracje AutoGluona, żeby wybrać model, który
trafi do dalszej części systemu, i ustalić, w jakiej postaci opuszcza on trening. Model musi dać
się wczytać bez ponownego treningu, w osobnym procesie, a razem z nim musi być zapisane jego
pochodzenie: dane, wersja kodu, metryki i wersje bibliotek.

Kryteria wyboru:

- F1 klasy `Churn = 1` na zbiorze testowym — metryka główna z karty projektu (15,71% odejść);
- czas predykcji jednego wiersza — model ma obsługiwać pojedyncze rekordy;
- rozmiar artefaktu;

Zbiór testowy ma 630 rekordów, w tym 99 odejść, więc różnica kilku rekordów mieści się
w zmienności wyniku. AutoGluon trenowaliśmy z limitem czasu, bez XGBoost i sieci neuronowych.

## Rozważone opcje

Model:

1. **LogisticRegression** — baseline ze Sprintów 1–2.
2. **`ag_fast`** — AutoGluon, preset `medium_quality`, limit treningu 60 s.
3. **`ag_quality`** — AutoGluon, preset `good_quality`, limit treningu 180 s.

Budowa pakietu:

- **A. Node w pipeline** — pakiet powstaje w tym samym `kedro run` co model.
- **B. Osobny skrypt** — pakiet budowany z już wytrenowanego modelu i danych jego runu z W&B.

## Decyzja

Model produkcyjny: **`ag_fast`**. Postać: pakiet z katalogiem modelu AutoGluona (`model/`)
i `metadata.json`, budowany w pipeline przez `build_model_package_node`
(`save_model_package()` z `src/asi_project/model_package.py`) i publikowany jako W&B Artifact
`churn-model` z aliasem `production`. Model wczytuje `load_model_package()` — tej samej funkcji
używają test w CI i smoke test.

Wyniki z W&B: tabela `model_comparison` z pełnego `kedro run` z 26.11.2026, z którego pochodzi
artefakt `churn-model:v0` (git `1333b71`). Zbiór testowy: 630 rekordów.

| Model | F1 | Wykryte odejścia (z 99) | Błędne alarmy | Predykcja 1 wiersza | Rozmiar | Pamięć po wczytaniu |
|---|---|---|---|---|---|---|
| LogisticRegression | 0,544 | 40 | 8 | 1,8 ms | 0,004 MB | 11,9 MB |
| **ag_fast** | **0,893** | **92** | **15** | **3,5 ms** | **36,8 MB** | **23,9 MB** |
| ag_quality | 0,901 | 91 | 12 | 6,3 ms | 79,5 MB | 24,0 MB |

Uzasadnienie:

- Baseline przeocza 59 z 99 odchodzących klientów, `ag_fast` tylko 7.
- `ag_quality` ma F1 wyższe o 0,008, czyli 2 błędy mniej na 630 rekordów (3 błędne alarmy
  mniej, ale 1 przeoczone odejście więcej). Jest za to 2,2 razy większy, a predykcja jednego
  wiersza trwa 1,8 razy dłużej. Pamięć po wczytaniu jest prawie taka sama.
- Wynik jest powtarzalny: we wszystkich czterech pełnych runach zapisanych w W&B macierze
  pomyłek obu modeli AutoGluona były identyczne, a `ag_fast` był 1,3–1,9 razy szybszy
  od `ag_quality`.
- Pakiet budowany w pipeline (opcja A) bierze model, metryki, Git SHA i fingerprint danych
  z jednego uruchomienia, więc `metadata.json` zawsze opisuje dokładnie ten model.

## Odrzucone alternatywy

- **LogisticRegression** — za niskie F1 (0,544) i recall (0,404). Jest szybsza i mniejsza,
  ale `ag_fast` też jest szybki i mały.
- **`ag_quality`** — 2 błędy mniej na 630 rekordów nie równoważą 2,2 razy większego artefaktu
  i wolniejszej predykcji.
- **Opcja B (osobny skrypt)** — trzeba ręcznie wskazać run W&B, metadane pochodzą z innego
  procesu niż model, a CI nie sprawdzałoby budowania pakietu.

## Konsekwencje

Ułatwia:

- kto potrzebuje modelu, pobiera `churn-model:production` i wczytuje go
  `load_model_package()`; nie musi znać numeru eksperymentu ani sposobu treningu;
- `metadata.json` podaje cechy wejściowe, target, klasę pozytywną, metryki, Git SHA,
  fingerprint danych i wersje bibliotek;
- CI przy każdym PR buduje pakiet z próbki i testuje zapis → odczyt → predykcję,
  a `scripts/smoke_model.py` sprawdza prawdziwy artefakt `production`.

Koszty:

- środowisko, które wczytuje model, potrzebuje `autogluon.tabular` 1.6.3 (wersja jest
  sprawdzana przy wczytaniu) oraz LightGBM i CatBoost; pakiet to katalog 36,8 MB, a nie
  jeden plik;
- każdy pełny `kedro run` trenuje modele od nowa (ok. 2,5 min) i publikuje nową wersję
  artefaktu z aliasem `production`, więc pełny run powinna uruchamiać jedna osoba;
- zmiana modelu produkcyjnego to zmiana `PRODUCTION_MODEL` w
  `src/asi_project/pipelines/data_science/pipeline.py` i aktualizacja tego ADR.

Otwarte kwestie:

- zgodnie z decyzją ze Sprintu 1 w danych zostało 300 duplikatów; 95 z 630 rekordów
  testowych ma identyczną kopię w zbiorze treningowym, więc F1 wszystkich modeli może być
  zawyżone.

## Historia

| Data | Zmiana |
|---|---|
| 26.11.2026 | utworzenie |
