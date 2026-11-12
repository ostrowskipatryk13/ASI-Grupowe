# ADR-1: Gdzie żyje przetwarzanie

- **Status:** proponowany
- **Sprint:** S2
- **Autor:** Patryk Ostrowski
- **Recenzent:** Wojciech Pyrzanowski 

## Kontekst

W Sprincie 2 trening przenieśliśmy z notebooka 02_baseline.ipynb do Kedro (kedro run). Od Sprintu 4 ten sam model będzie
używany przez API, które dostaje jeden rekord klienta (12 ALLOWED FEATURES z kontraktu iranian_churn),
bez targetu Churn i bez dostępu do reszty zbioru.

Trzeba więc ustalić, gdzie w kodzie żyje każdy z trzech rodzajów przetwarzania:
-tak żeby trening i predykcja przygotowywały dane identycznie
-predykcja nie wymagała targetu, a statystyki zbioru (mediany, średnie, kategorie) były liczone tylko
na zbiorze treningowym i trafiały do predykcji wyłącznie jako część modelu
-logika modelu ze sprintu 1 nie była kopiowana

Ograniczenia: 3150 rekordów, ~16% klasy pozytywnej, wszystkie kolumny liczbowe, 300 duplikatów

## Rozważone opcje

1. Opcja A — każdy rodzaj przetwarzania w osobnym miejscu: input normalization
   i training-only filtering jako funkcje w `data.py` wywoływane przez nody, model
   preprocessing wewnątrz sklearn `Pipeline`.
2. Opcja B — model preprocessing jako osobny node Kedro przed treningiem.
3. Opcja C - jedna funkcja łącząca input normalization i training-only filtering.

## Decyzja

Wybieramy opcję A.

| Wielkość | Wartość | Skąd pochodzi |
| Rekordy, na których dopasowuje się model preprocessing|2520(X_train) z 3150 | split_data_node, test_size: 0.2|
| Kolumny na wejściu modelu po model preprocessing |12 -> 29 | trained_model.pkl, get_feature_names_out() |

## Odrzucone alternatywy

Opcja B - przed splitem node uczyłby się także na 630 rekordach testowych, więc wynik testu przestałby być wiarygodny.
Po splicie zamiast jednego pliku z modelem byłyby dwa, które przy predykcji trzeba użyć razem i w dobrej kolejności.
Opcja C - filtrowanie potrzebuje targetu, a przy predykcji go nie ma.
Dlatego przy predykcji działa tylko normalize_input().
## Konsekwencje

Ułatwia:
- predykcja korzysta z tych samych funkcji co trenig bez kopiowania kodu
- dopasowany preprocessing i klasyfikator są w jednym miejscu, więc nie da się użyć klasyfikatora z niepasującym preprocessingiem.

Wymaga dyscypliny:
- data.py i model.py nie importują Kedro
- zmiana normalize_input() albo build_baseline_pipeline() zmienia model = po każdej takiej zmianie trzeba uruchomic kedro run
- nowy model preprocessing musi być zgodny ze sklearn, inaczej nie trafi do pipeline

## Historia

| Data | Zmiana |
| 12.11.2026 | utworzenie |
| |  |
