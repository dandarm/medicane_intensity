# Architettura iniziale e scostamenti dall'handoff

## Decisioni verificate nell'ambiente

L'handoff allegato è stato usato come traccia, ma il repository reale ha imposto
queste precisazioni:

1. Il downloader operativo è
   `/media/isacDisk1/Demetra/scripts/download_and_track_range.py` e usa la
   collection EUMETSAT `EO:EUM:DAT:MSG:MSG15-RSS`.
2. I prodotti RSS osservati sono distanziati di circa cinque minuti; Demetra
   normalizza il tempo al relativo slot `HH:MM` multiplo di cinque.
3. Demetra genera Airmass RGB, mentre il primo esperimento di optical flow
   richiede un canale fisico. Il sottoprogetto estrae quindi `IR_108` dagli stessi
   ZIP nativi e conserva la brightness temperature in `float32`.
4. Crop e orientamento non sono stati reimplementati: vengono importati dalla
   conversione locale già usata da Demetra, così gli array restano co-registrati
   con i suoi frame mediterranei.
5. Il repository Demetra contiene modifiche locali non correlate. Questo
   sottoprogetto non le modifica.

## Preprocessing

Le due immagini sono convertite a `float32`. NaN e maschera utente confluiscono
in una maschera valida comune. I percentili di clipping sono calcolati una sola
volta sull'unione dei valori validi dei due frame; la stessa trasformazione viene
applicata a entrambi. Questo evita una normalizzazione indipendente che potrebbe
inventare variazioni temporali di contrasto. Un Gaussian blur leggero è
configurabile.

## Algoritmo e quality control

Il backend iniziale è Farneback dense optical flow di OpenCV. I parametri sono
raccolti in `FarnebackConfig`, quindi in seguito è possibile aggiungere altri
backend mantenendo invariato `MotionField`.

Un vettore è valido solo quando:

- entrambi i frame sono osservabili;
- la deviazione standard locale supera la soglia di texture;
- lo spostamento è finito e sotto la soglia massima;
- il flow inverso, campionato nel punto di arrivo del flow diretto, chiude il
  ciclo entro la soglia forward/backward.

La confidence è separata dal campo e combina texture, range e consistenza
forward/backward. I pixel respinti assumono `NaN` in `dx`, `dy`, magnitude e
direzione, e zero nella confidence.

## Limiti attuali

- Il campo è cloud/image motion, non vento fisico né AMV.
- Divergenza e vorticità sono derivate in coordinate immagine e in unità
  pixel/pixel.
- Nessuna conversione in m/s: richiederà la geometria locale del pixel e la
  trasformazione del vettore nella base geografica.
- La soglia di texture e i parametri Farneback devono essere calibrati su più
  eventi e regimi nuvolosi.
- La validazione reale effettuata finora è visiva/statistica su una singola
  coppia; il test quantitativo disponibile è la traslazione sintetica.
- Il wrapper di download passa dal percorso completo Demetra, che produce anche
  Airmass RGB prima dell'estrazione IR 10.8.

## Prossimi passi sensati per la stima d'intensità

1. Costruire un catalogo di finestre per ciclone con intensità/pressione/vento e
   provenance verificabile.
2. Calcolare sequenze di motion field e aggregati radiali rispetto al centro del
   ciclone, senza confondere coordinate immagine e geografiche.
3. Implementare geolocalizzazione e velocità fisica locale, poi confrontare con
   AMV EUMETSAT dove disponibili.
4. Valutare robustezza temporale, gap, cambi di satellite e qualità del centro.
5. Solo dopo la validazione, usare i descrittori come input o target ausiliari di
   un modello per l'intensità.

