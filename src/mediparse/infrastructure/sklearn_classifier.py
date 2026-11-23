"""Klasický model v scikit-learn: TF-IDF s lineárním SVM a skops."""

from __future__ import annotations

import functools
import warnings
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Final, Literal

import skops.io as sio
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import FixedThresholdClassifier, TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from skops.io.exceptions import UntrustedTypesFoundException

from mediparse.domain.clinical_text import (
    TextNormalization,
    expand_diagnosis_acronyms,
    normalize_clinical_text,
    rewrite_diagnosis_variants,
)
from mediparse.domain.evaluation import Classification
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.sklearn_split import threshold_splits

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from pathlib import Path

    from mediparse.domain.identity import SubjectId
    from mediparse.domain.seed import Seed
    from mediparse.domain.tuning import ParamValue


EMPTY_VOCABULARY: Final = (
    "Trénovací texty neobsahují žádný token, slovník by byl prázdný."
)


class TfidfParameters(BaseModel):
    """Typové meze reprezentace TF-IDF na hranici sklearn."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", allow_inf_nan=False
    )

    ngram_min: PositiveInt = 1
    ngram_max: PositiveInt
    min_df: PositiveInt
    max_df: Annotated[float, Field(gt=0, le=1)]
    sublinear_tf: bool


class SvmParameters(TfidfParameters):
    """Parametry lineárního SVM nad sdílenou reprezentací TF-IDF."""

    analyzer: Literal["word", "char_wb"]
    c: PositiveFloat
    class_weight: Literal["balanced"] | None


def svm_vectorizer(
    params: SvmParameters, normalization: TextNormalization | None
) -> TfidfVectorizer:
    """Sestaví TF-IDF z validovaných parametrů ladění nebo tréninku.

    Returns:
        Nenatrénovaný vektorizér se společným předzpracováním.
    """
    return tfidf_vectorizer(
        normalization,
        analyzer=params.analyzer,
        ngram_range=(params.ngram_min, params.ngram_max),
        min_df=params.min_df,
        max_df=params.max_df,
        sublinear_tf=params.sublinear_tf,
    )


def svm_estimator(params: SvmParameters, seed: Seed) -> LinearSVC:
    """Sestaví lineární SVM z validovaných parametrů.

    Returns:
        Nenatrénovaný estimator se seedem běhu.
    """
    return LinearSVC(C=params.c, class_weight=params.class_weight, random_state=seed)


@dataclass(frozen=True)
class SklearnClassifier:
    """Natrénované TF-IDF a SVM s prahem z vnitřní CV po pacientech."""

    model: FixedThresholdClassifier

    @property
    def threshold(self) -> float:
        """Konečný práh na škále rozhodovací funkce uloženého modelu."""
        return float(self.model.threshold)

    def classify(self, texts: Sequence[str]) -> tuple[Classification, ...]:
        """Určí label a rozhodovací skóre lineárního SVM v jednom průchodu.

        Returns:
            Klasifikace v pořadí textů; label je ``predict``, skóre ``decision_function``.
        """
        listed = list(texts)
        labels = self.model.predict(listed)
        scores = self.model.decision_function(listed)
        return tuple(
            Classification(bool(label), float(score))
            for label, score in zip(labels, scores, strict=True)
        )


@dataclass(frozen=True)
class TrainedSvm(SklearnClassifier):
    """Výsledek tréninku s konvergencí všech vnitřních fitů pro manifest."""

    converged: bool


ESTIMATOR: Final = "sklearn TfidfVectorizer+LinearSVC+FixedThresholdClassifier"


def _require_svm_shape(model: object, path: Path) -> None:
    """Odmítne archiv mimo podporovanou rodinu klasického modelu.

    Raises:
        InvalidInputError: Model nemá podporovaný tvar.
    """
    # Predikce používá estimator_, parametr estimator nemusí odpovídat fitu.
    frozen = getattr(model, "estimator_", None)
    pipeline = frozen.estimator if isinstance(frozen, FrozenEstimator) else None
    steps = pipeline.steps if isinstance(pipeline, Pipeline) else []
    if not isinstance(model, FixedThresholdClassifier) or [
        (name, type(step)) for name, step in steps
    ] != [("tfidf", TfidfVectorizer), ("svm", LinearSVC)]:
        msg = f"Soubor modelu {path} nemá tvar {ESTIMATOR}."
        raise InvalidInputError(msg)


def tfidf_preprocessor(
    text: str, acronyms: Mapping[str, str], variants: Mapping[str, str]
) -> str:
    """Normalizace, rozvinutí zkratek, přepis variant a snížení písmen před tokenizací TF-IDF.

    Returns:
        Normalizovaný text malými písmeny, zkratky rozvinuté a varianty i překlepy přepsané.
    """
    # Rozvinutí rozlišuje velikost písmen, proto jde před .lower(). Zkratky i
    # varianty běží po normalizaci, aby podtržítka v ___COPD či ___T2DM
    # neblokovala hranici slova.
    expanded = expand_diagnosis_acronyms(normalize_clinical_text(text), acronyms)
    # Vlastní preprocessor vypíná lowercase vectorizeru, snížení písmen proto nese
    # tato funkce. Bez něj by CAST a Cast byly dva příznaky.
    return rewrite_diagnosis_variants(expanded, variants).lower()


# Volby TF-IDF zůstávají keyword-only, aby stávající volání zachovala výchozí chování.
def tfidf_vectorizer(  # ruff: ignore[too-many-arguments]
    normalization: TextNormalization | None,
    *,
    analyzer: Literal["word", "char_wb"] = "word",
    ngram_range: tuple[int, int] = (1, 1),
    min_df: int = 1,
    max_df: float = 1.0,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Vectorizer pro trénink, ladění i měření v sešitu EDA.

    Args:
        normalization: Slovníky sdíleného preprocessoru, nebo None pro jeho vypnutí.
            Preprocessor se uplatní i u znakového analyzátoru char_wb.
        analyzer: Typ analyzátoru word nebo char_wb; výchozí word zachovává slovník
            tréninku a EDA.
        ngram_range: Nejmenší a největší délka n-gramu; výchozí (1, 1) zachovává
            unigramový slovník tréninku a EDA.
        min_df: Nejmenší počet dokumentů s příznakem.
        max_df: Největší podíl dokumentů s příznakem.
        sublinear_tf: Zda použít logaritmickou frekvenci místo počtu výskytů.

    Returns:
        Nenatrénovaný vectorizer; bez normalizace je to ablace bez preprocessoru.
    """
    preprocessor = (
        None
        if normalization is None
        # Holý dict, ne Mapping ani pydantic objekt: skops uloží dict, jiný typ
        # by vyžadoval další důvěryhodný typ.
        else functools.partial(
            tfidf_preprocessor,
            acronyms=dict(normalization.acronyms),
            variants=dict(normalization.variants),
        )
    )
    return TfidfVectorizer(
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        max_df=max_df,
        preprocessor=preprocessor,
        analyzer=analyzer,
        ngram_range=ngram_range,
    )


def ensure_tokens(vectorizer: TfidfVectorizer, texts: Sequence[str]) -> None:
    """Odmítne texty, ze kterých analyzátor nevytvoří žádný příznak.

    Raises:
        InvalidInputError: Trénovací slovník by byl prázdný.
    """
    analyzer = vectorizer.build_analyzer()
    require_tokens(analyzer(text) for text in texts)


def require_tokens(documents: Iterable[Sequence[str]]) -> None:
    """Odmítne dokumenty bez jediného tokenu.

    Raises:
        InvalidInputError: Trénovací slovník by byl prázdný.
    """
    if not any(documents):
        raise InvalidInputError(EMPTY_VOCABULARY)


# Seznam platí při načtení, ne při uložení: skops model uloží vždy a při načtení
# pustí funkci jen z tohoto seznamu. Přejmenování funkce proto rozbije načítání
# dříve uložených modelů. Oba slovníky nese functools.partial, který musí být
# v seznamu také; funkci uvnitř partialu skops kontroluje zvlášť.
TRUSTED_FUNCTIONS: Final = (
    "functools.partial",
    "mediparse.infrastructure.sklearn_classifier.tfidf_preprocessor",
)


@dataclass(frozen=True)
class LinearSvmTrainer:
    """Trénink TF-IDF s lineárním SVM."""

    parameters: Mapping[str, ParamValue]
    seed: Seed
    normalization: TextNormalization | None
    threshold_folds: int

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[bool],
        subjects: Sequence[SubjectId],
    ) -> TrainedSvm:
        """Zvolí práh ve vnitřní CV a refitne pipeline na celém tréninku.

        Returns:
            Klasifikátor s uložitelným prahem a konvergencí všech fitů.

        Raises:
            InvalidInputError: Neplatné parametry SVM nebo nepoužitelný trénovací slovník.
        """
        try:
            params = SvmParameters.model_validate(self.parameters)
            vectorizer = svm_vectorizer(params, self.normalization)
            ensure_tokens(vectorizer, texts)
            tuned = TunedThresholdClassifierCV(
                Pipeline([
                    ("tfidf", vectorizer),
                    ("svm", svm_estimator(params, self.seed)),
                ]),
                scoring="f1",
                cv=threshold_splits(labels, subjects, self.threshold_folds, self.seed),
            )
            converged = fit_converged(lambda: tuned.fit(list(texts), list(labels)))
        except ValueError as error:
            msg = f"Parametry nebo trénovací slovník SVM nejsou platné: {error}"
            raise InvalidInputError(msg) from error
        # TTCV už refitoval celou pipeline; FrozenEstimator zabrání druhému fitu.
        model = FixedThresholdClassifier(
            FrozenEstimator(tuned.estimator_),
            threshold=float(tuned.best_threshold_),
            response_method="decision_function",
        ).fit(list(texts), list(labels))
        return TrainedSvm(model, converged)


@dataclass(frozen=True)
class SkopsModelFile:
    """Model uložený ve formátu skops, který se načítá bez důvěry v cizí typy a funkce."""

    path: Path

    def save(self, classifier: SklearnClassifier) -> None:
        """Uloží model s prahem a nadřazený adresář založí.

        Bajty souboru se mezi běhy mohou lišit: skops pojmenovává členy ZIPu
        podle id() objektů a ukládá čas zápisu. Stejný model proto dokládají
        stejné predikce včetně skóre, nikoli SHA-256 souboru.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        sio.dump(classifier.model, self.path)

    def load(self) -> SklearnClassifier:
        """Načte model; z cizích typů důvěřuje jen funkcím ze ``TRUSTED_FUNCTIONS``.

        Returns:
            Klasifikátor s načteným modelem a prahem.

        Raises:
            InvalidInputError: Soubor modelu neexistuje, je poškozený nebo
                obsahuje typ mimo důvěryhodné nebo nemá podporovaný tvar.
        """
        try:
            model = sio.load(self.path, trusted=list(TRUSTED_FUNCTIONS))
        except FileNotFoundError as error:
            msg = f"Soubor modelu {self.path} neexistuje."
            raise InvalidInputError(msg) from error
        except (zipfile.BadZipFile, UntrustedTypesFoundException) as error:
            msg = f"Soubor modelu {self.path} nelze bezpečně načíst: {error}"
            raise InvalidInputError(msg) from error
        _require_svm_shape(model, self.path)
        return SklearnClassifier(model)


def fit_converged(fit: Callable[[], object]) -> bool:
    """Fitne estimator a zachová ostatní varování jako diagnostiku volajícího.

    Returns:
        Zda všechny fity estimatoru konvergovaly.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fit()
    # record=True zachytí i jiné kategorie; jejich diagnostika nesmí zmizet.
    for item in caught:
        if not issubclass(item.category, ConvergenceWarning):
            warnings.warn_explicit(
                item.message, item.category, item.filename, item.lineno
            )
    return not any(issubclass(item.category, ConvergenceWarning) for item in caught)
