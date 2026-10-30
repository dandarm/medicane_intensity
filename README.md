# Medicane intensity

Sottoprogetto per studiare descrittori dinamici utili alla stima dell'intensità
dei cicloni mediterranei da sequenze MSG-SEVIRI RSS. La prima componente
implementata stima il **moto apparente delle strutture nell'immagine** sul canale
fisico `IR_108` (10.8 µm). Non è un vento atmosferico e non è un prodotto AMV
EUMETSAT: mancano ancora height assignment, trasformazione geografica e dati NWP.

## Stato iniziale

- download RSS nominalmente ogni 5 minuti riusando, senza duplicarlo,
  `/media/isacDisk1/Demetra/scripts/download_and_track_range.py`;
- estrazione della brightness temperature `IR_108` dai ZIP nativi EUMETSAT;
- crop e orientamento identici a Demetra tramite le sue utility geometriche;
- dense optical flow Farneback (OpenCV);
- maschera di qualità per validità, texture, range dello spostamento e controllo
  forward/backward;
- magnitude, divergenza e vorticità come **proxy nello spazio immagine**;
- figura QA, serializzazione NPZ e test sintetico della convenzione dei segni.

## Ambiente

L'ambiente già presente e verificato è:

```bash
/home/isac/miniconda3/envs/videomae/bin/python
```

Per rendere importabile il package senza installarlo:

```bash
export PYTHONPATH=/media/isacDisk1/medicane_intensity/src
```

In alternativa, dall'ambiente desiderato:

```bash
python -m pip install -e /media/isacDisk1/medicane_intensity
```

## Download a 5 minuti ed estrazione IR 10.8

Le credenziali rimangono esclusivamente nelle variabili d'ambiente:

```bash
export EUMETSAT_CONSUMER_KEY='...'
export EUMETSAT_CONSUMER_SECRET='...'

/home/isac/miniconda3/envs/videomae/bin/python \
  scripts/download_seviri_ir108.py \
  --start '2026-03-26 10:00' \
  --end '2026-03-26 11:00'
```

Il wrapper chiama il downloader di Demetra con `--skip_inference`, conserva i
prodotti nativi sotto `data/downloads/range_*/raw_eumetsat/` ed estrae i file
`data/ir108/ir108_YYYYMMDD_HHMM.npz`. Poiché il downloader Demetra non ha ancora
una modalità `raw-only`, crea anche i PNG Airmass RGB; non vengono usati per il
motion field.

Per estrarre nuovamente gli ZIP già scaricati nel root di questo progetto:

```bash
/home/isac/miniconda3/envs/videomae/bin/python \
  scripts/download_seviri_ir108.py \
  --start ignored --end ignored --skip-download
```

## Demo su due frame consecutivi

```bash
MPLBACKEND=Agg /home/isac/miniconda3/envs/videomae/bin/python \
  scripts/demo_motion.py \
  data/ir108/ir108_20260326_1000.npz \
  data/ir108/ir108_20260326_1005.npz
```

Lo script stampa mediana, percentili e frazione valida, poi salva:

- il campo (`dx`, `dy`, modulo, direzione, qualità, confidence e metadati) in NPZ;
- una figura con IR 10.8, quiver, magnitude, vorticità e divergenza proxy.

Per sovrapporre coste, campo vettoriale colorato e centro del ciclone:

```bash
MPLBACKEND=Agg /home/isac/miniconda3/envs/videomae/bin/python \
  scripts/demo_motion.py \
  data/ir108/ir108_20260317_1200.npz \
  data/ir108/ir108_20260317_1205.npz \
  --coastlines \
  --center-lat 33.29 --center-lon 16.28 \
  --zoom-radius-x 260 --zoom-radius-y 190 \
  --quiver-step 8 --arrow-exaggeration 4 \
  --output-dir outputs/jolina_20260317_1200
```

Le coste usano la medesima proiezione geostazionaria e lo stesso dominio
mediterraneo di Demetra. Il fattore `--arrow-exaggeration` modifica soltanto la
lunghezza grafica; colore e legenda continuano a riportare pixel per 5 minuti.

## API Python

```python
from medicane_intensity.motion import estimate_motion

field = estimate_motion(
    image_t,
    image_t1,
    timestamp_t,
    timestamp_t1,
    channel="IR_108",
    valid_mask=valid_mask,  # opzionale
)
```

Convenzione: `dx > 0` indica moto verso destra (colonne crescenti), `dy > 0`
verso il basso (righe crescenti). `direction_deg = atan2(dy, dx)`. Gli
spostamenti sono in pixel per frame e `dt_seconds` è mantenuto separato. Non si
assume una distanza chilometrica costante per pixel.

## Test

```bash
PYTHONPATH=src /home/isac/miniconda3/envs/videomae/bin/python \
  -m unittest discover -s tests -v
```

Il test sintetico applica una traslazione nota di `dx=+3`, `dy=-2` e verifica
esplicitamente segni, accuratezza e intervallo di 300 secondi.

## Struttura

```text
src/medicane_intensity/
  motion/
    preprocessing.py
    optical_flow.py
    diagnostics.py
    visualization.py
  seviri/
    ir108.py
scripts/
  download_seviri_ir108.py
  demo_motion.py
tests/
  test_motion.py
```

Le scelte architetturali, i limiti e i prossimi passi sono descritti in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
