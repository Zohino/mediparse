"""Preregistrované pravidlo non-inferiority a cenového srovnání modelů."""

import math
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from mediparse.domain.costs import LatencyPlan
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.seed import Seed


class Verdict(StrEnum):
    """Tři výsledky posouzení non-inferiority modelu A vůči modelu B."""

    NON_INFERIOR = "non_inferior"
    INFERIOR = "inferior"
    INCONCLUSIVE = "inconclusive"


class DecisionRule(BaseModel):
    """Preregistrované pravidlo pro rozdíl macro-F1 modelu A minus model B.

    Primární hypotéza srovnává nejlepší laděnou klasiku (A) s nejlepším
    transformerem (B), oba vybrané na validaci, nad celou zprávou v hlavním
    řádku EN. Macro-F1 je průměr F1 kladné třídy přes pět diagnóz ze sloučených
    out-of-fold predikcí všech K foldů evaluace. Práh obou ramen se volí
    stejně, vnitřní CV po pacientech jen na tréninkové části příslušného foldu.
    Oboustranný percentilový CI s hladinou ``confidence`` vzniká párovým
    cluster bootstrapem přes pacienty na týchž zprávách obou ramen.

    Jde o jedno primární srovnání bez korekce na multiplicitu. Strip
    diagnostických sekcí zůstává ablací: rozdíl s CI se reportuje popisně,
    bez verdiktu. Čeština a další transformery jsou také popisná srovnání.
    Rozdíly F1 s CI po jednotlivých diagnózách se reportují bez verdiktu.

    Výsledek hypotézy podle verdiktu celé zprávy a ``is_cheaper``:

    | Verdikt | Levnější | Hypotéza |
    |---|---|---|
    | NON_INFERIOR | True | podpořena |
    | NON_INFERIOR | False | nepodpořena: non-inferiorní, ale ne levnější |
    | INFERIOR | libovolně | vyvrácena |
    | INCONCLUSIVE | libovolně | nerozhodnuto |

    SD macro-F1 transformeru se měří přes ``seed_runs`` seedy na jednom foldu.
    Je-li vzdálenost rozdílu od hranice minus delta ostře menší než
    ``seed_sensitivity_sd_multiple`` krát SD, výsledek se označí jako citlivý
    na seed. Značka verdikt nemění a uvádí se u každého výsledku hypotézy.
    U každého výsledku se uvede i výhrada věrohodnosti, pokud macro-F1
    transformeru ostře nepřevýší neladěný TF-IDF + LR na téže sadě.
    Jde o kontrolu věrohodnosti transformeru, nikoli assay sensitivity.

    Rodina A ε-dominuje B pro ε = ``margin``, pokud je A non-inferiorní
    týmž testem, není dražší v latenci p95 ani energii inference a v alespoň
    jedné ose je ostře levnější. Paretova hranice je popisná; primárním
    verdiktem zůstává jediná dvojice modelů výše.

    Levnější znamená alespoň ``min_latency_ratio`` krát vyšší latenci p95
    nebo ``min_inference_energy_ratio`` krát vyšší energii inference
    transformeru na referenčním notebooku ``reference_cpu`` a ``reference_gpu``.
    Klasika běží na CPU, transformer na GPU, dávka je 1 a měření po zahřátí
    řídí ``latency``. Latence zahrnuje předzpracování, tokenizaci a všechna
    okna zprávy; vedle p95 se reportuje medián. Hranice 10krát bude podle
    odhadu nejspíš splněna: vymezuje levnější model, není silným testem.

    Energie inference v J = CPU-sekundy procesu krát ``cpu_watts_per_core``
    plus energie GPU z NVML. Obě ramena měří totéž okno dávkové klasifikace
    týchž testových zpráv foldu 0 po warmupu, mezi started a finished
    v ModelInference. Benchmark celého kroku Snakemake se nepoužije.
    CPU-sekundy zahrnují všechna vlákna. NVML měří celé zařízení včetně
    klidového odběru u obou ramen a klid se ani jednomu neodečítá.

    Výkon na fyzické jádro 2,8125 W = Processor Base Power 45 W / 16 jader
    (6P + 8E + 2LP) Core Ultra 9 185H; Intel, specifikace SKU 236849,
    ověřeno 2026-11-15. Maximum Turbo Power 115 W se nepoužije. Metoda
    per-core TDP vychází z Green Algorithms (Lannelongue a kol., 2021,
    arXiv:2007.07610). Směr odchylky odhadu vůči skutečné spotřebě není znám:
    jednovláknová zátěž v turbu bere víc než 45/16 W, souběh dvou vláken
    jednoho P-jádra méně. Trénink na rozdílném HW se hodnotí jen popisně: GPU-h, CPU-h
    a kWh včetně ladění do cenového verdiktu nevstupují.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    margin: Annotated[float, Field(gt=0, lt=1)]
    min_latency_ratio: Annotated[float, Field(ge=1)]
    min_inference_energy_ratio: Annotated[float, Field(ge=1)]
    confidence: Annotated[float, Field(gt=0, lt=1)]
    bootstrap_resamples: Annotated[int, Field(ge=1000)]
    min_evaluation_notes: PositiveInt
    seed: Seed
    seed_runs: Annotated[int, Field(ge=2)]
    seed_sensitivity_sd_multiple: Annotated[float, Field(gt=0)]
    latency: LatencyPlan
    reference_cpu: Annotated[str, Field(min_length=1)]
    reference_gpu: Annotated[str, Field(min_length=1)]
    cpu_watts_per_core: Annotated[float, Field(gt=0)]

    def verdict(self, ci_low: float, ci_high: float, evaluation_notes: int) -> Verdict:
        """Určí non-inferioritu z mezí CI s ostrou hranicí minus delta.

        Args:
            ci_low: Dolní mez CI rozdílu macro-F1 modelu A minus model B.
            ci_high: Horní mez téhož CI.
            evaluation_notes: Počet společně evaluovaných zpráv.

        Returns:
            Prokázaná, vyvrácená nebo nerozhodnutá non-inferiorita.

        Raises:
            InvalidInputError: Neplatné meze CI nebo příliš málo zpráv.
        """
        if not -1 <= ci_low <= ci_high <= 1:
            msg = "Neplatné meze CI: musí být v rozsahu [-1, 1] a dolní mez nesmí překročit horní."
            raise InvalidInputError(msg)
        if evaluation_notes < self.min_evaluation_notes:
            msg = f"Evaluace vyžaduje alespoň {self.min_evaluation_notes} zpráv."
            raise InvalidInputError(msg)
        if ci_low > -self.margin:
            return Verdict.NON_INFERIOR
        if ci_high < -self.margin:
            return Verdict.INFERIOR
        return Verdict.INCONCLUSIVE

    def is_cheaper(self, latency_ratio: float, energy_ratio: float) -> bool:
        """Ověří dosažení alespoň jednoho prahu úspory klasického modelu.

        Args:
            latency_ratio: Poměr latence p95 na zprávu transformer ku klasice.
            energy_ratio: Poměr energie inference transformer ku klasice.

        Returns:
            Zda alespoň jeden poměr dosahuje preregistrovaného prahu.

        Raises:
            InvalidInputError: Některý poměr není konečný nebo je záporný.
        """
        _validate_nonnegative_finite(latency_ratio, energy_ratio)
        return (
            latency_ratio >= self.min_latency_ratio
            or energy_ratio >= self.min_inference_energy_ratio
        )

    def is_seed_sensitive(self, difference: float, seed_sd: float) -> bool:
        """Označí citlivost na seed podle vzdálenosti rozdílu od hranice NI.

        Args:
            difference: Rozdíl macro-F1 klasika minus transformer.
            seed_sd: SD macro-F1 transformeru přes seedy na jednom foldu.

        Returns:
            Zda je vzdálenost ostře menší než preregistrovaný násobek SD.

        Raises:
            InvalidInputError: Rozdíl není konečný v [-1, 1] nebo SD není
                konečné a nezáporné.
        """
        if not -1 <= difference <= 1:
            msg = "Rozdíl macro-F1 musí být konečný v rozsahu [-1, 1]."
            raise InvalidInputError(msg)
        _validate_nonnegative_finite(seed_sd)
        return (
            abs(difference + self.margin) < self.seed_sensitivity_sd_multiple * seed_sd
        )

    def inference_energy_joules(self, cpu_seconds: float, gpu_joules: float) -> float:
        """Sečte odhad energie CPU a měření GPU ve společném okně inference.

        Args:
            cpu_seconds: CPU-sekundy procesu za okno přes všechna vlákna.
            gpu_joules: Energie celého zařízení z NVML včetně klidového odběru.

        Returns:
            Odhad energie inference v joulech na referenčním notebooku.

        Raises:
            InvalidInputError: Některý vstup není konečný nebo je záporný.
        """
        _validate_nonnegative_finite(cpu_seconds, gpu_joules)
        return cpu_seconds * self.cpu_watts_per_core + gpu_joules

    @staticmethod
    def is_credible(transformer_f1: float, baseline_f1: float) -> bool:
        """Ověří, že transformer ostře překonává neladěný TF-IDF + LR.

        Args:
            transformer_f1: Macro-F1 transformeru na společné evaluační sadě.
            baseline_f1: Macro-F1 neladěného TF-IDF + LR na téže sadě.

        Returns:
            Zda transformer splňuje preregistrovanou kontrolu věrohodnosti.

        Raises:
            InvalidInputError: Některá macro-F1 není konečná v [0, 1].
        """
        if not (0 <= transformer_f1 <= 1 and 0 <= baseline_f1 <= 1):
            msg = "Macro-F1 transformeru i baseline musí být konečná v rozsahu [0, 1]."
            raise InvalidInputError(msg)
        return transformer_f1 > baseline_f1

    def dominates(
        self,
        ci_low: float,
        ci_high: float,
        evaluation_notes: int,
        latency_ratio: float,
        energy_ratio: float,
    ) -> bool:
        """Určí ε-dominanci rodiny A vůči B s ε rovným okraji non-inferiority.

        Args:
            ci_low: Dolní mez CI rozdílu macro-F1 modelu A minus model B.
            ci_high: Horní mez téhož CI.
            evaluation_notes: Počet společně evaluovaných zpráv.
            latency_ratio: Poměr latence p95 na zprávu modelu B ku modelu A.
            energy_ratio: Poměr energie inference modelu B ku modelu A.

        Returns:
            Zda je A non-inferiorní, nedražší v obou osách a v jedné levnější.

        Raises:
            InvalidInputError: Neplatné meze CI, příliš málo zpráv nebo
                některý poměr není konečný a nezáporný.
        """
        _validate_nonnegative_finite(latency_ratio, energy_ratio)
        return (
            self.verdict(ci_low, ci_high, evaluation_notes) is Verdict.NON_INFERIOR
            and min(latency_ratio, energy_ratio) >= 1
            and max(latency_ratio, energy_ratio) > 1
        )


def _validate_nonnegative_finite(*values: float) -> None:
    if not all(math.isfinite(value) and value >= 0 for value in values):
        msg = "Naměřené hodnoty nebo poměry musí být konečné a nezáporné."
        raise InvalidInputError(msg)
