# Automatická extrakce informací z lékařských zpráv pomocí NLP

Repozitář k bakalářské práci zaměřené na NLP v lékařských zprávách. Obsahuje
experimenty s klasifikací nálezů (pozitivní/negativní), NER (věk, pohlaví, komorbidity)
a porovnání klasických ML metod a transformerových modelů na veřejných datových sadách.

## Smoketest

Smoketest projde celou cestu od přípravy dat přes trénink po evaluaci na
syntetických zprávách. Běží v kontejneru bez sítě, nepotřebuje MIMIC ani
žádné credentials a trvá několik sekund po sestavení image.

### Požadavky

- git a repozitář naklonovaný přes `git clone` (archiv ZIP nestačí, run
  manifest potřebuje commit)
- podman nebo docker (použije se podman, jinak docker; přebít jde proměnnou
  `CONTAINER_ENGINE`)

### Spuštění

Linux, macOS, WSL a Git Bash (sh, bash, zsh):

```sh
./smoketest.sh
```

Windows, PowerShell i cmd:

```sh
powershell -NoProfile -ExecutionPolicy Bypass -File smoketest.ps1
```

Skript sestaví image, předá mu commit a příznak rozpracovaného stromu, v
kontejneru bez sítě pustí přípravu dat, trénink, evaluaci a načtení uloženého
modelu a na konci vypíše tabulku metrik a ukázku predikce.

### Výstupy

- `build/smoketest/`: data, model, predikce, metriky, ukázka predikce uloženého
  modelu (`demo.parquet`) a run manifest
- `logs/smoketest/`: logy jednotlivých kroků

### Testy

```sh
podman run --rm mediparse pytest
```
