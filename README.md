# Cable Detection con Mask R-CNN

Questo repository contiene un notebook per l'addestramento e la validazione di un modello di **Instance Segmentation** basato su **Mask R-CNN (ResNet50 FPN)**, ottimizzato specificamente per il rilevamento e l'estrazione delle traiettorie dei cavi in immagini.

L'obiettivo del progetto non si limita alla semplice segmentazione dei pixel, ma si estende alla **stima matematica della retta** che rappresenta il cavo (attraverso le sue coordinate polari $\rho$ e $\theta$), utile in applicazioni di computer vision per l'analisi industriale o la robotica.

## 🚀 Caratteristiche Principali

- **Segmentazione e Bounding Box**: Rilevamento preciso della posizione del cavo utilizzando `torchvision`.
- **Estrazione della Traiettoria (Thinning e Hough)**:
  - Algoritmo di assottigliamento (thinning morfologico con OpenCV) per estrarre lo scheletro della maschera segmentata.
  - Trasformata di Hough (`cv2.HoughLines`) per calcolare la retta dominante del cavo in coordinate polari (`rho`, `theta`).
- **Metrica LDS (Line Detection Score)**: Una metrica personalizzata utilizzata per valutare e guidare l'addestramento. Combina:
  - Precision e Recall (AP50, AR50) standard del formato COCO.
  - Penalità basate sull'errore di stima dell'angolo ($\theta$) e della distanza dall'origine ($\rho$).
- **Data Augmentation Progressiva**: Vengono applicati tre diversi regimi di augmentation durante l'addestramento:
  - *Strong* (rotazioni, flip orizzontali/verticali, color jitter) per le prime epoche.
  - *Light* (flip orizzontali e rotazioni leggere) per le epoche centrali.
  - *Off* per il fine-tuning conclusivo.
- **Learning Rate Scheduler & Early Stopping**: Utilizzo di `ReduceLROnPlateau` per massimizzare la metrica LDS e interruzione anticipata dell'addestramento in caso di stallo per prevenire l'overfitting.

## 🛠️ Requisiti di Sistema e Librerie

Il codice è stato sviluppato per girare in ambienti con accelerazione GPU (es. Kaggle, Google Colab).
Dipendenze principali (installate automaticamente nel notebook):
- `torch==2.1.2`
- `torchvision==0.16.2`
- `numpy`
- `opencv-python-headless`
- `pycocotools`
- `shapely`

## 📂 Struttura del Dataset

Il progetto si aspetta un dataset di immagini e annotazioni formattato secondo lo standard **COCO (Common Objects in Context)**, ma con un'aggiunta fondamentale:
- Ogni annotazione contiene il campo standard `segmentation` (per le maschere o poligoni).
- È presente un campo custom `polar_coordinates` contenente la _ground truth_ per `rho` e `theta` del cavo.

Nel notebook viene generato e "patchato" dinamicamente un set di validazione `gt_val_patched.json` dividendo i dati di train in proporzione (85% train, 15% validation).

## 🧠 Pipeline del Modello

1. **Preprocessing**: Le immagini vengono trasformate in tensori. I bounding box troppo piccoli o invalidi vengono scartati.
2. **Forward Pass**: Il modello Mask R-CNN restituisce Bounding Box, Scores, Labels e Maschere (Soft Masks).
3. **Postprocessing**:
   - Vengono mantenute solo le predizioni con score superiore a una soglia e area minima accettabile.
   - La maschera viene binarizzata e passata alla funzione `thinning()`.
   - Vengono calcolati `rho` e `theta` per ogni maschera.
   - Le maschere vengono convertite in formato RLE (Run-Length Encoding) per l'integrazione con Pycocotools.
4. **Valutazione (COCOeval + LDS)**: Durante la fase di validazione, le maschere predette sono confrontate con la Ground Truth per calcolare l'IoU, ed estrarre la discrepanza angolare per il calcolo del *Line Detection Score*.

## 📝 Risultati e Output

Il notebook salverà all'interno dell'ambiente di lavoro:
- Le predizioni sul validation set: `val_preds_epoch_X.json`
- Il checkpoint dell'ultima epoca: `maskrcnn_cable_ckpt.pth`
- Il modello migliore in assoluto in base al LDS: `maskrcnn_cable_best.pth`
- Il modello finale post-addestramento: `maskrcnn_cable_final.pth`

Questi file possono essere usati per fare inferenza (nel set di test) o riprendere l'addestramento in un momento successivo.
