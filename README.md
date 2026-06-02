# System Monitorowania Modelu ML - Środowisko Uruchomieniowe

Informacja: Niniejszy plik README stanowi techniczne uzupełnienie analitycznej i teoretycznej części pracy, która została szczegółowo przedstawiona w odrębnym [dokumencie PDF w tym repozytorium](https://github.com/yaps-dev/mlp-sin-regression/blob/main/Ewelina%20To%C5%82wi%C5%84ska%20-%20System%20Monitorowania%20Modelu%20ML.pdf). Skupia się on wyłącznie na instrukcji wdrożenia i obsługi środowiska eksperymentalnego.

Niniejsze repozytorium zawiera konfigurację środowiska Docker Compose dla systemu monitorowania modelu uczenia maszynowego. Środowisko integruje aplikację serwującą model (FastAPI) z bazą danych (PostgreSQL) oraz pełnym stosem monitoringu (Prometheus, Grafana, cAdvisor).
## 📌 Wymagania wstępne

Aby uruchomić projekt, upewnij się, że masz zainstalowane w swoim systemie:

* **Docker**
* **Docker Compose** (lub wtyczkę `docker compose` w nowszych wersjach Dockera)

*Uwaga: Zwróć uwagę, że usługa `app` buduje się z kontekstu katalogu wyżej (`context: ..`) i oczekuje modelu w `../artifacts/`.*

## 🚀 Uruchomienie środowiska

1. Otwórz terminal w katalogu monitoring-system, gdzie znajduje się plik `docker-compose.yml`.
2. Zbuduj i uruchom kontenery w tle za pomocą polecenia:

```bash
docker compose up -d --build

```

3. Sprawdź, czy wszystkie usługi wstały poprawnie:

```bash
docker compose ps

```

## 🌐 Dostępne usługi i porty

Po pomyślnym uruchomieniu, poszczególne komponenty systemu są dostępne pod adresami `http://localhost:<PORT>`.

| Usługa | Port | Opis                                                                                                                            | Poświadczenia (User / Password) |
| --- | --- |---------------------------------------------------------------------------------------------------------------------------------| --- |
| **App (ML API)** | `8000` | Główne REST API serwujące model ML i wystawiające endpoint z metrykami `/metrics`. dokumentacja api dostępna pod ścieżką`/docs` | *Brak uwierzytelnienia* |
| **Grafana** | `3000` | Wizualizacja metryk i dashboardy                                                                                                | `admin` / `admin` (włączony dostęp anonimowy dla roli Admin) |
| **Prometheus** | `9090` | System zbierania metryk                                                                                                         | *Brak uwierzytelnienia* |
| **cAdvisor** | `8088` | Statystyki zasobów sprzętowych kontenerów                                                                                       | *Brak uwierzytelnienia* |
| **pgAdmin** | `5480` | Panel zarządzania bazą danych PostgreSQL                                                                                        | `admin@domain.com` / `password` |
| **PostgreSQL** | `5432` | Baza danych do persystencji predykcji                                                                                           | `username` / `password` (Baza: `ml_data`) |

## ⚡ Generowanie obciążenia (Load Generator)
W katalogu src/load-generator znajduje się dedykowany skrypt load-generator.py. Został on przygotowany i skonfigurowany pod zaprezentowane środowisko uruchomieniowe.

Narzędzie to pozwala na wygenerowanie różnych wariantów obciążenia dla API serwującego model. Dzięki niemu możesz łatwo zasymulować określony ruch sieciowy oraz zjawiska prowadzące do degradacji modelu (np. powtarzalne indukowanie dryfu danych i dryfu koncepcji), co umożliwia przetestowanie mechanizmów monitoringu w praktyce i obserwację zmian na dashboardach Grafany.

## 🛑 Zatrzymywanie środowiska

Aby zatrzymać działające kontenery, nie usuwając zapisanych danych (predykcji w bazie), uruchom:

```bash
docker compose down

```

Jeśli chcesz całkowicie usunąć środowisko **wraz z danymi bazy PostgreSQL** (wyczyszczenie wolumenów), uruchom:

```bash
docker compose down -v

```

## 📝 Przydatne polecenia diagnostyczne

Podgląd logów z usługi aplikacji ML (FastAPI):

```bash
docker compose logs -f app

```