# dev_V5.5 – Documentation de développement (core + orchestrateur `superior`)

> **But du document**
> Décrire pour un·e dev :
>
> * le **cœur V5.5** (prepare / train / evaluate, multi-corpus, idéologie, configs),
> * la **couche d’orchestration expérimentale** (`superior`),
> * la **politique RAM** et les mécanismes avancés (multi-corpus, cross-dataset, early-stop, OOM policy),
>   en un seul document cohérent.

Ce document remplace **`dev_V4.md`** (core) et **`dev_V5.md`** (orchestrateur) comme **référence unique**.

---

## 0. Contexte et vision V5.5

### 0.1. Héritage V4.x

Le projet V4.x a posé le cœur actuel :

* 3 scripts principaux :

  * `core_prepare.py` : TEI → TSV + formats spaCy / sklearn / HF,
  * `core_train.py` : entraînement des familles de modèles,
  * `core_evaluate.py` : évaluation, métriques, rapports.
* une couche config centrée sur :

  * `configs/profiles/*.yml` (profils d’analyse),
  * `configs/common/*.yml` (corpora, hardware, balance, models),
  * `configs/label_maps/*.yml` (idéologie, acteurs).
* un pipeline **multi-familles / multi-vues**, mais initialement **mono-dataset**.

### 0.2. Extensions V4.2 / V5

Deux axes majeurs :

1. **Multi-corpus / multi-dataset (V4.2)**

   * `dataset_id`, `data.corpus_ids`, `merge_mode`, `source_field`, `analysis.compare_by`.
   * Capacité à combiner `web1`, `asr1`, `web2`, etc., en un dataset logique unique, avec métriques globales et par sous-groupe.

2. **Orchestrateur expérimental (V5 / `superior`)**

   * Lecture d’une **config d’expérience** YAML (axes, grilles, scheduler),
   * génération d’un **plan de runs**,
   * orchestration de `prepare/train/evaluate` via des **sous-processus**,
   * gestion de la **RAM** et du **parallélisme**,
   * hooks d’analyse (courbes, rapports).

### 0.3. Objectif V5.5

V5.5 vise à :

* **stabiliser définitivement** le cœur multi-corpus,
* **unifier** la doc core + orchestrateur,
* définir une **politique RAM** claire :

  * soft limits (hardware + scheduler) en premier,
  * hard limit (`--max-ram-mb`) uniquement en option,
* intégrer les mécanismes avancés :

  * multi-corpus / multi-vue,
  * cross-dataset,
  * early-stop / OOM policy,
  * hooks d’analyse (`curves`, `report_markdown`).

---

## 1. Architecture globale

### 1.1. Vue d’ensemble (core + orchestrateur)

```mermaid
flowchart LR
    subgraph RAW["data/raw/<corpus_id>/"]
        A[corpus.xml (TEI)]
    end

    subgraph CORE_PREPARE["core_prepare.py"]
        B[Extraction TEI → docs]
        C[Mapping idéologie]
        D[Équilibrage]
        E[Split train/job]
        F[TSV + formats]
    end

    subgraph INTERIM["data/interim/<dataset_id>/<view>/"]
        T1[train.tsv]
        T2[job.tsv]
        MV[meta_view.json]
    end

    subgraph PROCESSED["data/processed/<dataset_id>/<view>/"]
        S1[spacy shards]
        S2[sklearn TSV]
        H1[hf JSONL]
        MF[meta_formats.json]
    end

    subgraph MODELS["models/<dataset_id>/<view>/"]
        MCHECK[check/.../meta_model.json]
        MSPA[spacy/<model_id>/]
        MSKL[sklearn/<model_id>/]
        MHF[hf/<model_id>/]
    end

    subgraph REPORTS["reports/<dataset_id>/<view>/"]
        R1[metrics.json]
        R2[metrics_by_<field>.json]
        R3[classification_report.txt]
        R4[meta_eval.json]
    end

    subgraph SUPERIOR["superior_orchestrator.py"]
        P[plan.tsv]
        RUNS[runs.tsv]
        L[logs/run_*.log]
        MG[metrics_global.tsv]
        PL[plots/*.png]
        REP[report.md]
    end

    A --> CORE_PREPARE --> INTERIM --> PROCESSED --> MODELS --> REPORTS
    SUPERIOR -->|orchestration| CORE_PREPARE
    SUPERIOR --> MODELS
    SUPERIOR --> REPORTS
```

---

## 2. Cœur V5.5 – core_prepare / core_train / core_evaluate

### 2.1. Principes

* **Core minimal & générique** : 3 scripts, interface stable CLI (profil + overrides).
* **Tout le métier dans les configs** : corpora, hardware, modèles, équilibres, idéologie définis en YAML.
* **Multi-*** par construction :

  * multi-corpus (via `corpora.yml` + `dataset_id`),
  * multi-vues (ideology_global, left/right intra, etc.),
  * multi-familles (`check`, `spacy`, `sklearn`, `hf`),
  * multi-modèles par famille (`models.yml`).

### 2.2. Organisation des fichiers côté core

* `configs/common/` :

  * `corpora.yml` : définitions des corpus TEI (id, chemin, langue…).
  * `hardware.yml` : presets (small, lab, …).
  * `balance.yml` : stratégies d’équilibrage.
  * `models.yml` : inventaire de modèles par famille.
* `configs/label_maps/` :

  * `ideology.yml`, `ideology_actors.yml`, `ideology_global.yml`, etc.
* `configs/profiles/*.yml` :

  * profils d’analyse (ideo_quick, ideo_full, crawl_*, etc.).
* `scripts/core/` :

  * `core_prepare.py`, `core_train.py`, `core_evaluate.py`, `core_utils.py`.

### 2.3. Résolution du profil (profil + overrides)

Entrée standard :

```bash
python scripts/core/core_prepare.py --profile ideo_quick --override key=val ...
python scripts/core/core_train.py   --profile ideo_quick ...
python scripts/core/core_evaluate.py --profile ideo_quick ...
```

Ou via Makefile :

```bash
make run STAGE=pipeline PROFILE=ideo_quick
```

Tout passe par `resolve_profile_base(profile_name, overrides)` dans `core_utils.py`.

Fonctions clés :

1. Charger `configs/profiles/<profile>.yml`.
2. Résoudre :

   * `corpus_id` / `dataset_id`,
   * `data.corpus_ids`, `merge_mode`, `source_field`,
   * `analysis.compare_by`.
3. Mêler avec `corpora.yml`, `hardware.yml`, `balance.yml`, `models.yml`.
4. Appliquer les `--override key=val`.
5. Retourner un dict `params` commun aux 3 scripts, contenant au minimum :

```python
params = {
  "profile": ...,
  "dataset_id": ...,
  "corpora": [...],       # liste des corpus_cfg
  "merge_mode": ...,
  "source_field": ...,
  "analysis": {...},      # dont compare_by
  "view": ...,
  "families": [...],
  "models_spacy": [...],
  "models_sklearn": [...],
  "models_hf": [...],
  "hardware": {...},      # dont max_train_docs_*
  "balance": {...},       # stratégie + preset
  "train_prop": ...,
  "seed": ...,
  ...
}
```

---

### 2.4. Multi-corpus & `dataset_id`

#### 2.4.1. Concepts

* **`dataset_id`** : identifiant logique de l’ensemble d’analyse (répertoires `data/interim`, `data/processed`, `models`, `reports`). Ex. `web1`, `asr1`, `web1_asr1`.
* **`data.corpus_ids`** : liste de corpus sources (`web1`, `asr1`, …) déclarés dans `corpora.yml`.
* **`merge_mode`** :

  * `"single"` : corpus unique (mode V4),
  * `"merged"` : fusion brute,
  * `"juxtaposed"` : fusion + métriques par groupe (`metrics_by_<field>.json`),
  * `"separate"` : chaque corpus traité séparément (utilisé via orchestrateur).
* **`source_field`** : nom de la colonne (et meta) qui identifie la source (`corpus_id`, `modality`, …).
* **`analysis.compare_by`** : liste de champs sur lesquels on veut des métriques groupées.

#### 2.4.2. Profil multi-corpus typique

```yaml
profile: ideo_quick_web1_asr1
description: >
  Fusion web1+asr1, vue ideology_global, analyse juxtaposée.

dataset_id: web1_asr1

data:
  corpus_ids: [web1, asr1]
  merge_mode: juxtaposed
  source_field: corpus_id

view: ideology_global

analysis:
  compare_by:
    - corpus_id

families: [check, spacy, sklearn]
models_spacy: [spacy_cnn_quick]
models_sklearn: [tfidf_svm_quick]
hardware_preset: small
train_prop: 0.8
balance_strategy: oversample
balance_preset: parity
```

---

### 2.5. `core_prepare.py` – TEI → TSV + formats (multi-corpus)

#### 2.5.1. Rôle

* Parser les TEI (`corpus.xml`),
* appliquer les mappings idéologiques,
* filtrer / équilibrer,
* splitter train/job,
* sérialiser TSV + formats modèles,
* gérer le cas multi-corpus (`corpus_ids` / `dataset_id`).

#### 2.5.2. Chemins

Avec `dataset_id` résolu :

```python
dataset_id = params["dataset_id"]    # ex: "web1" ou "web1_asr1"
view = params["view"]                # ex: "ideology_global"

interim_dir   = data/interim/<dataset_id>/<view>/
processed_dir = data/processed/<dataset_id>/<view>/
```

#### 2.5.3. Lecture multi-TEI

Pseudocode :

```python
docs = []
source_field = params.get("source_field", "corpus_id")
for corpus_cfg in params["corpora"]:
    tei_path = corpus_cfg["corpus_path"]
    corpus_id_src = corpus_cfg["corpus_id"]

    for doc in iter_tei_docs(tei_path, params):
        row = build_row(doc, params)  # id, text, label, etc.
        row[source_field] = corpus_id_src
        docs.append(row)
```

#### 2.5.4. Équilibrage & split

* Équilibrage via `balance_strategy` (`none`, `cap_docs`, `cap_tokens`, `alpha_total`, `class_weights`, …).
* Split stratifié `train` / `job` selon `train_prop`.
* Écrit :

```text
data/interim/<dataset_id>/<view>/train.tsv
data/interim/<dataset_id>/<view>/job.tsv
data/interim/<dataset_id>/<view>/meta_view.json
```

`meta_view.json` contient :

* `dataset_id`,
* `source_corpora` (liste),
* `source_field`,
* stats label / split / équilibre.

#### 2.5.5. Formats modèles

En fonction de `families` :

* **spaCy** : shards DocBin pour éviter les DocBin géants (`E870`) :

  ```text
  data/processed/<dataset_id>/<view>/spacy/train/part-0001.spacy
  data/processed/<dataset_id>/<view>/spacy/job/part-0001.spacy
  ```

* **sklearn** : réutilise TSV (liens + meta).

* **HF** : JSONL / TSV convertible (prévu pour entraînement stocké).

`meta_formats.json` décrit les chemins de chaque format par famille.

---

### 2.6. `core_train.py` – entraînement des familles

#### 2.6.1. Rôle & dispatch

* Résoudre `params`,
* fixer la seed globale (optionnel),
* appliquer `hardware` (`max_train_docs_*`, threads BLAS),
* pour chaque famille active (`families`), entraîner les modèles déclarés dans `models.yml`.

Pseudocode simplifié :

```python
params = resolve_profile_base(...)
hw = params["hardware"]

families = params["families"]
models_to_train = []

if "check" in families:
    models_to_train.append(("check", "check_default"))
if "spacy" in families:
    for mid in params["models_spacy"]:
        models_to_train.append(("spacy", mid))
...
for family, model_id in models_to_train:
    train_family_model(family, model_id, params)
```

#### 2.6.2. Famille `check`

* Pseudo-modèle :

  * calcule des stats sur `train.tsv` (distribution des labels, nb docs),
  * crée un `meta_model.json` de contrôle dans `models/<dataset_id>/<view>/check/check_default/`.

#### 2.6.3. Famille `spacy`

* Entrée : shards DocBin (train/job).
* Utilise un cfg spaCy externe (`configs/spacy/<model_id>.cfg`).
* Entraîne dans :

```text
models/<dataset_id>/<view>/spacy/<model_id>/
  model-best/
  meta_model.json
```

* Respecte `hardware.max_train_docs_spacy` (tronquage si nécessaire).

#### 2.6.4. Famille `sklearn`

* Entrée : `train.tsv`.
* Modèles définis dans `models.yml` (`tfidf_svm_quick`, etc.).
* Gère **class weights** si `balance_strategy=class_weights`.

#### 2.6.5. Famille `hf`

* Même pattern : `models_hf` + `models.yml`,
* Entrée : JSONL/TSV transformé,
* Gère `hardware.max_train_docs_hf` + batch size / max_length.

---

### 2.7. `core_evaluate.py` – évaluation & métriques

#### 2.7.1. Rôle

* Charger les modèles entraînés,
* évaluer sur `job.tsv` (et formats spacy/hf),
* produire :

```text
reports/<dataset_id>/<view>/<family>/<model_id>/metrics.json
reports/.../classification_report.txt
reports/.../meta_eval.json
reports/.../metrics_by_<field>.json (si compare_by)
```

#### 2.7.2. Multi-groupe (`analysis.compare_by`)

Si le profil définit :

```yaml
analysis:
  compare_by:
    - corpus_id
```

Alors `core_evaluate` :

* groupby sur `corpus_id` dans `job.tsv`,
* calcule des métriques pour chaque groupe,
* écrit `metrics_by_corpus_id.json` :

```json
{
  "web1": { "accuracy": ..., "macro_f1": ..., ... },
  "asr1": { "accuracy": ..., "macro_f1": ..., ... }
}
```

Le mécanisme est généralisable à d’autres champs (ex : `modality`).

#### 2.7.3. Cross-dataset (V5.5)

Extension introduite en V5.5 :

* possibilité de séparer :

  * `dataset_id_for_models` : d’où viennent les modèles,
  * `dataset_id_for_eval` : de quel dataset provient `job.tsv`.

Dans `core_evaluate` :

```python
dataset_id_for_models = params["dataset_id"]
dataset_id_for_eval = params.get("eval_dataset_id", dataset_id_for_models)
```

* les modèles sont chargés depuis `models/<dataset_id_for_models>/<view>/...`,
* `job.tsv` est lu depuis `data/interim/<dataset_id_for_eval>/<view>/job.tsv`.

---

## 3. Sous-système idéologie

### 3.1. De `ideology.yml` aux vues dérivées

Pipeline conceptuel :

1. **Référentiel conceptuel** : `configs/label_maps/ideology.yml`

   * catégories globales (far_right, right, center, left, far_left, etc.),
   * clusters intra (droite/gauche),
   * tags d’acteurs/domaine.

2. **Squelette d’acteurs** :

   * `scripts/pre/make_ideology_skeleton.py` :

     * scanne TEI,
     * produit `ideology_actors.yml` (squelette) + `actors_counts_<corpus>.tsv`.

3. **Dérivation des vues** :

   * `scripts/pre/derive_ideology_from_yaml.py` :

     * fusionne `ideology.yml` + squelette,
     * génère :

       * `ideology_actors.yml` final,
       * `ideology_global.yml`,
       * `ideology_left_intra.yml`,
       * `ideology_right_intra.yml`, etc.

4. **Profils** :

   * les profils (ideo_quick, ideo_full…) référencent :

     * `label_map` : un des YAML dérivés,
     * `view` : `ideology_global`, `left_intra`, `right_intra`, etc.

---

## 4. Orchestrateur expérimental V5.5 – `superior`

### 4.1. Nomenclature & fichiers

Historiquement, la spec parlait d’`exp_orchestrator` / `experiments/`.
En V5.5, l’implémentation réelle s’appelle **`superior`** :

* `scripts/superior/superior_orchestrator.py`
* `scripts/superior/run_single.py`
* `configs/superior/*.yml` (config d’expériences)
* `superior/<exp_id>/` :

  * `plan.tsv` : plan de runs,
  * `runs.tsv` : statut / méta de runs,
  * `logs/run_*.log`,
  * `metrics_global.tsv`,
  * `plots/*.png`,
  * `report.md`.

> **NOTE** : dans les docs plus anciennes,
> `exp_orchestrator` = `superior_orchestrator.py` et
> `experiments/<exp_id>/` = `superior/<exp_id>/`.

### 4.2. Rôle

`superior` :

* lit une **exp config** YAML,
* génère un **plan** (combinations d’axes × répétitions),
* exécute chaque run via `run_single` (qui appelle `make run`),
* contrôle le **parallélisme** et la **pression RAM**,
* applique éventuellement des politiques `oom_policy` et `early_stop`,
* déclenche les hooks d’analyse à la fin (courbes, rapport).

---

## 5. Config d’expérience `configs/superior/*.yml`

### 5.1. Structure générale

Exemple type (simplifié) :

```yaml
exp_id: ideo_balancing_sweep
description: >
  Étude de l'impact des stratégies d'équilibrage
  sur web1 et web1+asr1, avec ideo_quick.

base:
  profile: ideo_quick
  stage: pipeline        # pipeline | prepare | train | evaluate
  make_vars:
    HARDWARE_PRESET: small
    TRAIN_PROP: 0.8
  overrides:
    ideology.view: ideology_global

axes:
  - name: dataset
    type: choice
    values:
      - label: web1_only
        overrides:
          dataset_id: web1
          data.corpus_ids: [web1]
          data.merge_mode: single

      - label: web1_asr1_juxt
        overrides:
          dataset_id: web1_asr1
          data.corpus_ids: [web1, asr1]
          data.merge_mode: juxtaposed
          analysis.compare_by: [corpus_id]

  - name: balance_strategy
    type: choice
    values:
      - label: no_balance
        make_vars:
          BALANCE_STRATEGY: none
      - label: oversample_parity
        make_vars:
          BALANCE_STRATEGY: oversample
          BALANCE_PRESET: parity

  - name: family_model
    type: choice
    values:
      - label: sklearn_svm
        make_vars:
          FAMILY: sklearn
        overrides:
          families: [sklearn]
          models_sklearn: [tfidf_svm_quick]

      - label: spacy_cnn
        make_vars:
          FAMILY: spacy
        overrides:
          families: [spacy]
          models_spacy: [spacy_cnn_quick]

grid:
  mode: cartesian

run:
  repeats: 1
  seed_strategy: per_run   # fixed | per_run
  base_seed: 42

scheduler:
  parallel: 1
  max_ram_gb: 14
  resource_classes:
    spacy: heavy
    sklearn: light
    hf: heavy
  weights:
    light: 1
    medium: 2
    heavy: 4
  max_weight: 4

safety:
  enable_hard_ram_limit: false
  hard_limit_mb: null

oom_policy:
  on_oom: "skip"           # "skip" | "backoff" | "stop"
  backoff_factor: 0.5

early_stop:
  enabled: true
  min_accuracy: 0.30
  min_macro_f1: 0.25
  apply_to_families: [spacy, hf]

analysis_hooks:
  after_experiment:
    - type: curves
      metrics: [accuracy, macro_f1]
      x_axis: TRAIN_PROP
      group_by: [family, dataset_id]

    - type: report_markdown
      path: superior/${exp_id}/report.md
```

### 5.2. Sémantique

* **base** :

  * `profile` : nom du profil core,
  * `stage` : stage core par défaut (`pipeline`, `train`, etc.),
  * `make_vars` : variables Make communes à tous les runs,
  * `overrides` : overrides (clé=val) appliqués à `resolve_profile_base`.

* **axes** :

  * `type: choice` + `values[*].label` :

    * `make_vars` : variables Make spécifiques à cette valeur,
    * `overrides` : overrides spécifiques (`data.corpus_ids`, `analysis.compare_by`, etc.).

* **grid.mode** :

  * `cartesian` = produit cartésien des axes.

* **run** :

  * `repeats` : nb de répétitions,
  * `seed_strategy` :

    * `fixed` : même `SEED` pour tous,
    * `per_run` : `SEED = base_seed + run_index`.

* **scheduler** :

  * `parallel` : nb de runs simultanés max,
  * `max_ram_gb` : budget RAM global **soft**,
  * `resource_classes`, `weights`, `max_weight` :

    * heuristique pour ne pas lancer trop de runs “heavy” ensemble.

* **safety** :

  * `enable_hard_ram_limit` + `hard_limit_mb` :

    * s’ils sont activés, `superior` transmettra `--max-ram-mb` à `run_single` (hard kill per-run),
    * par défaut **false** → pas de hard kill.

* **oom_policy / early_stop** :

  * cf. section 7.

* **analysis_hooks** :

  * liste de hooks à lancer après l’expérience : courbes, rapport, agrégats.

---

## 6. Structures internes (V5.5)

### 6.1. `RunSpec`

```python
@dataclass
class RunSpec:
    run_id: str                 # ex: "run_000123"
    exp_id: str
    profile: str
    stage: str                  # "pipeline" | "prepare" | "train" | "evaluate"
    make_vars: Dict[str, str]   # PROFILE, STAGE, CORPUS_ID, FAMILY, ...
    overrides: Dict[str, Any]   # pour OVERRIDES="k1=v1 k2=v2 ..."
    repeat_index: int
    axis_values: Dict[str, str] # { "dataset": "web1_asr1_juxt", ... }
    resource_class: str         # "light" | "medium" | "heavy"
```

### 6.2. `ExpConfig` & `SchedulerConfig`

```python
@dataclass
class OomPolicy:
    on_oom: Literal["skip", "backoff", "stop"] = "skip"
    backoff_factor: float = 0.5

@dataclass
class EarlyStopConfig:
    enabled: bool = False
    min_accuracy: float = 0.0
    min_macro_f1: float = 0.0
    apply_to_families: List[str] = field(default_factory=list)

@dataclass
class SchedulerConfig:
    parallel: int = 1
    max_ram_gb: Optional[float] = None  # soft limit
    resource_classes: Dict[str, str] = field(default_factory=dict)
    weights: Dict[str, int] = field(default_factory=dict)
    max_weight: int = 4
    approx_ram_per_class: Dict[str, float] = field(default_factory=dict)  # V5.5
    oom_policy: Optional[OomPolicy] = None
    early_stop: Optional[EarlyStopConfig] = None

@dataclass
class SafetyConfig:
    enable_hard_ram_limit: bool = False
    hard_limit_mb: Optional[int] = None

@dataclass
class ExpConfig:
    exp_id: str
    description: str
    base_profile: str
    base_stage: str
    base_make_vars: Dict[str, str]
    base_overrides: Dict[str, Any]
    axes: List[AxisConfig]
    grid_mode: str
    repeats: int
    seed_strategy: str
    base_seed: int
    scheduler: SchedulerConfig
    safety: SafetyConfig
    analysis_hooks: AnalysisHooksConfig
```

---

## 7. Exécution d’un run – `run_single.py`

### 7.1. CLI

Exemple :

```bash
python -m scripts.superior.run_single \
  --exp-id ideo_balancing_sweep \
  --run-id run_0001 \
  --profile ideo_quick \
  --stage pipeline \
  --make-var CORPUS_ID=web1 \
  --make-var FAMILY=spacy \
  --override ideology.view=ideology_global \
  --override dataset_id=web1_asr1 \
  --override data.corpus_ids=[web1,asr1] \
  --log-path superior/ideo_balancing_sweep/logs/run_0001.log
  # éventuellement :
  # --max-ram-mb 12000
```

### 7.2. Construction de la commande core

Pseudocode :

```python
cmd = ["make", "run", f"STAGE={stage}", f"PROFILE={profile}"]
for k, v in make_vars.items():
    cmd.append(f"{k}={v}")

if overrides:
    overrides_str = " ".join(f"{k}={v}" for k, v in overrides.items())
    cmd.append(f'OVERRIDES={overrides_str}')
```

Puis :

* `subprocess.Popen(cmd, ...)`,
* log stdout/stderr vers `log_path`.

### 7.3. Hard limit RAM (optionnel)

Si `--max-ram-mb` est fourni **et** `psutil` dispo :

* boucle de monitoring : toutes les X secondes, lire `rss`,
* si `rss > max_ram_mb` :

  * logguer dans stdout + fichier de log,
  * `terminate()` puis `kill()` si besoin,
  * retourner un code spécifique (ex. `99`), interprété comme `status="oom"`.

> En V5.5, ce mécanisme est **désactivé par défaut**.
> Il n’est activé que si `safety.enable_hard_ram_limit` est `true` dans `exp_config`
> **et** que `hard_limit_mb` est défini.

---

## 8. Scheduler – `superior_orchestrator.py`

### 8.1. CLI

```bash
python -m scripts.superior.superior_orchestrator \
  --exp-config configs/superior/exp_ideo_balancing_sweep.yml \
  --parallel 2 \
  --max-ram-gb 14 \
  --max-runs 100 \
  --resume \
  --dry-run
```

* `--parallel` / `--max-ram-gb` : peuvent surcharger `scheduler` de la config.
* `--max-runs` : limite de debug.
* `--resume` : ne relance pas les runs déjà `success` dans `runs.tsv`.
* `--dry-run` : génère seulement `plan.tsv`.

### 8.2. Génération du plan

Étapes :

1. Parser `exp_config.yml` → `ExpConfig`.
2. Pour chaque combinaison d’axes (`grid.mode=cartesian`), pour chaque `repeat` :

   * construire `make_vars` (base + axes),
   * construire `overrides` (base + axes),
   * déterminer `resource_class` (selon `family` / `models_*`),
   * créer un `RunSpec`.
3. Écrire `plan.tsv` avec toutes les infos (`run_id`, `axis_values_json`, `make_vars_json`, `overrides_json`, `resource_class`…).

### 8.3. Boucle de scheduling (V5.5)

Pseudo-code de haut niveau :

```python
pending = load_plan(...)           # liste de RunSpec
active = {}                        # run_id -> ProcessHandle
completed = {}                     # run_id -> RunResult
while pending or active:
    # Terminer les runs finis
    for run_id, proc in list(active.items()):
        if proc.poll() is not None:
            result = collect_result(proc)
            completed[run_id] = result
            update_runs_tsv(run_id, result)
            apply_oom_policy_if_needed(result, config, pending)

            del active[run_id]

    # Lancer de nouveaux runs si capacité
    while len(active) < scheduler.parallel and pending:
        next_run = pick_next_run(pending, scheduler, completed)
        if not has_soft_ram_budget_for(next_run, active, scheduler):
            break
        proc = launch_run(next_run, safety)
        active[next_run.run_id] = proc

    sleep(1.0)

run_analysis_hooks(exp_config, runs.tsv)
```

### 8.4. Soft limit RAM (V5.5)

Fonction `has_soft_ram_budget_for(next_run, active, scheduler)` :

* utilise `scheduler.approx_ram_per_class` :

  * ex. `{"light": 2.0, "medium": 4.0, "heavy": 8.0}` (en Go).
* calcule :

```python
current_ram = sum(
    scheduler.approx_ram_per_class[resource_class(run)]
    for run in active.values()
)
needed_ram = scheduler.approx_ram_per_class[next_run.resource_class]
if scheduler.max_ram_gb is not None:
    return current_ram + needed_ram <= scheduler.max_ram_gb
else:
    return True
```

* si faux → on attend qu’un run se termine avant d’en lancer un autre,
* **aucun kill** sur cette base : c’est purement une limite de pilotage.

### 8.5. Politique OOM (`oom_policy`)

Lorsqu’un run se termine avec `status="oom"` (code 99) :

* `on_oom="skip"` :

  * marquer les `RunSpec` futurs correspondant à la même combinaison d’axes/famille comme `skippés` (par ex. `status="skipped_oom"` dans `runs.tsv`).
* `on_oom="backoff"` (version minimale) :

  * marquer et logguer, skip pour l’instant,
  * version future : réinjecter un override modifié (ex : réduire `max_train_docs_*`).
* `on_oom="stop"` :

  * en cas de OOM, marquer tous les runs restants comme `aborted_oom_policy` et sortir.

### 8.6. Early-stop (`early_stop`)

Après chaque run `evaluate` réussi :

1. Lire `metrics.json` via `metrics_path`.
2. Si `family ∈ apply_to_families` **et**
   `accuracy < min_accuracy` **et** `macro_f1 < min_macro_f1` :

   * logguer un warning dans `superior/<exp_id>/logs/early_stop.log`,
   * enrichir `runs.tsv` avec :

     * `quality_ok = false`,
     * `quality_flags = "low_accuracy,low_macro_f1"`.

Version V5.5 minimale = **log-only**.
Une version ultérieure peut décider de réduire `repeats` ou couper certaines branches d’axes.

---

## 9. Hooks d’analyse (`analysis_hooks`)

### 9.1. Hook `curves`

* Lit `runs.tsv` + `metrics_global.tsv` (ou re-chargement via `metrics_path`).
* Pour chaque métrique `metric` dans `metrics` :

  * construit un dataset `(x_axis, valeur)` pour chaque groupe (`group_by`).
  * trace via `matplotlib` des courbes `metric` vs `x_axis`,
  * sauvegarde dans `superior/<exp_id>/plots/<metric>__vs__<x_axis>.png`.

### 9.2. Hook `report_markdown`

* Synthétise `runs.tsv` (success/failed/oom, temps, seeds, axes),
* met en avant :

  * meilleurs runs par famille / dataset / split,
  * anomalies (OOM, qualité faible),
  * liens vers les graphes (hook `curves`),
* écrit un Markdown lisible dans `superior/<exp_id>/report.md`.

---

## 10. Scénarios avancés

### 10.1. Multi-corpus juxtaposé : web1 + asr1

* Profil `ideo_quick_web1_asr1` (cf. §2.4.2),
* `merge_mode=juxtaposed`, `source_field=corpus_id`, `compare_by=[corpus_id]`,
* core :

  * crée `data/interim/web1_asr1/...`,
  * `metrics_by_corpus_id.json` par famille/modèle.
* orchestrateur :

  * un axe `dataset` qui compare `web1_only` vs `web1_asr1_juxt`,
  * hook `curves` qui trace `accuracy` / `macro_f1` vs `TRAIN_PROP` pour chaque dataset.

### 10.2. Cross-dataset : train web1 / eval asr1

* `core_evaluate` supporte `eval_dataset_id` (cf. §2.7.3).
* dans `exp_config` :

```yaml
axes:
  - name: dataset_pair
    type: choice
    values:
      - label: train_web1_eval_asr1
        overrides:
          dataset_id: web1
          cross_dataset.train_on: web1
          cross_dataset.eval_on: asr1
```

* orchestrateur :

  * génère un run `train` pour `web1`,
  * puis un run `evaluate` avec `dataset_id=web1` + `eval_dataset_id=asr1`.

---

## 11. Politique RAM V5.5 – Guidelines

1. **Hard limit (`--max-ram-mb`)** :

   * exclusivement optionnelle,
   * à activer via `safety.enable_hard_ram_limit=true`,
   * à réserver aux scénarios de grid massifs / non supervisés.

2. **Soft limits (recommandé)** :

   * calibrer `hardware.yml` + `max_train_docs_*` pour qu’un **run seul** tienne sur la machine,
   * laisser `scheduler.parallel` bas (1–2) sur laptop,
   * définir `scheduler.max_ram_gb` et `approx_ram_per_class` raisonnables.

3. **Surveiller `runs.tsv` et `max_rss_mb` (si exposé)** pour ajuster :

   * réduire `max_train_docs_*`,
   * diminuer `TRAIN_PROP` sur les profils très gourmands,
   * abaisser `parallel` ou `max_weight`.

---

## 12. Backlog résiduel (post-V5.5)

Quelques points restent volontairement ouverts pour itérations ultérieures :

* Implémenter réellement la stratégie `oom_policy.backoff` (modification dynamique de `max_train_docs_*`).
* Donner une sémantique complète à `early_stop` (réduction de `repeats`, pruning d’axes).
* Étendre la famille HF (multi-GPU ultérieur, gradient accumulation, etc.).
* Ajout de tests automatiques :

  * unitaires sur `resolve_profile_base` multi-corpus,
  * intégration core (`prepare/train/evaluate`) minimal,
  * tests orchestrateur (plan, scheduler, hooks).
