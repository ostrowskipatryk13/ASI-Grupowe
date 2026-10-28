# Architektura systemu

Ten dokument opisuje architekturę systemu zespołu. W Sprincie 1 powstaje diagram v0, który
będzie aktualizowany w kolejnych sprintach.

## Architektura docelowa (scaffold)

Tak wygląda system po sześciu sprintach — na poziomie warstw. To mapa odpowiedzialności,
a nie diagram do skopiowania: diagram v0 (niżej) rysujecie sami.

Relacje między warstwami (opis tekstowy, nie diagram):

- training: dane → pipeline Kedro → artefakt modelu
- serving: UI → API → model (artefakt wczytany przez API)
- state: API → persistence
- deployment: API + UI → Cloud Run

| Warstwa | Odpowiedzialność | Sprint | Miejsce w repozytorium |
|---|---|---|---|
| Dane i kontrakt | źródło danych, kontrakt, input normalization, training-only filtering | S1 | `data/01_raw/`, `src/asi_project/datasets.yaml`, `src/asi_project/data.py` |
| Pipeline | odtwarzalne przetwarzanie i trening od surowych danych | S2 | `conf/`, `src/asi_project/pipelines/` |
| Eksperymenty i artefakt | porównanie runów, wybór modelu, pakiet modelu ładowalny poza treningiem | S3 | W&B, pakiet modelu, `docs/model_card.md` |
| API | kontrakt usługi, walidacja na granicy, model wczytywany raz | S4 | `src/asi_project/api/`, `docker/` |
| Persistence | zapis predykcji poza procesem aplikacji (tryby `sql` / `disabled`) | S4 → S5 → S6 | repository w API, Postgres w compose |
| UI | formularz jako klient API | S5 | `ui/`, `docker/`, `docker-compose.yml` |
| Chmura | wdrożenie, sekret, logi, release | S6 | `deploy/`, Cloud Run, Secret Manager, Cloud Logging |

## Diagram v0 (Sprint 1)

Narysujcie, jak — według waszej obecnej wiedzy — będzie wyglądał system na koniec projektu.
Diagram v0 nie musi być poprawny w szczegółach; ma pokazać wasze rozumienie przepływu.
Sama lista warstw z sekcji wyżej nie wystarcza — diagram pokazuje wasz system i wasze decyzje.

Na diagramie zaznaczcie co najmniej:

- skąd pochodzą dane i gdzie żyje kontrakt danych;
- gdzie działa każdy z trzech typów przetwarzania: input normalization, training-only
  filtering, model preprocessing;
- gdzie powstaje model i jak trafia tam, gdzie wykonuje się predykcja;
- kto lub co wysyła żądanie predykcji i co dostaje w odpowiedzi;
- gdzie przechowywany jest stan (np. zapis predykcji);
- które elementy już istnieją po Sprincie 1, a które są planowane.

Diagram umieśćcie w bloku `mermaid` w sekcji poniżej (GitHub renderuje Mermaid w Markdown)
albo jako obraz w `docs/` z linkiem w tej sekcji. Usuńcie znaczniki TODO, gdy diagram będzie
gotowy — sprawdza to `python scripts/check_dod.py`.

<!-- DIAGRAM-V0:START -->
```mermaid
flowchart TD
    subgraph S1 [Sprint 1: Dane i Kontrakt - Istnieje]
        RawData[(Surowe Dane / CSV)] --> Contract[Kontrakt Danych & Dataset Schema]
        Contract --> InpNorm[Input Normalization]
        Contract --> TrainFilter[Training-only Filtering]
    end

    subgraph S2_S3 [Sprint 2 & 3: Pipeline i Model - Planowane]
        InpNorm --> ModelPrep[Model Preprocessing]
        TrainFilter --> ModelPrep
        ModelPrep --> TrainPipeline[Pipeline Treningowy / Kedro]
        TrainPipeline --> ModelArtifact[(Artefakt Modelu / W&B)]
    end

    subgraph S4_S5 [Sprint 4 & 5: Serving, API i UI - Planowane]
        ModelArtifact --> API[FastAPI Service]
        Client[Użytkownik / UI] -->|HTTP Request| API
        API -->|Predykcja| Client
    end

    subgraph Persistence [Stan i Zapis - Planowane]
        API -->|Zapis predykcji| DB[(PostgreSQL / Persistence)]
    end

    style S1 fill:#d4edda,stroke:#28a745,stroke-width:2px
    style S2_S3 fill:#fff3cd,stroke:#ffc107,stroke-width:2px
    style S4_S5 fill:#f8d7da,stroke:#dc3545,stroke-width:2px
    style Persistence fill:#e2e3e5,stroke:#6c757d,stroke-width:2px
```

Diagram przedstawia docelowy, wielowarstwowy przepływ danych i komponentów systemu w wersji v0. Po Sprincie 1 wdrożono kluczowe reguły kontraktu, normalizacji oraz filtrowania danych wejściowych. W kolejnych etapach projekt zostanie rozbudowany o automatyczny pipeline treningowy, rejestr modeli oraz warstwę serwowania opartą na REST API i interfejsie użytkownika. Całość domyka wydzielona warstwa trwałości stanu służąca do zapisu predykcji.
<!-- DIAGRAM-V0:END -->

## Historia zmian architektury

| Sprint | Zmiana | Powiązany ADR |
|---|---|---|
| S1 | diagram v0 | — |
