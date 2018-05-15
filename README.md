# Určení typu a směru zbraně v obrazové scéně
* EN - Determination of Gun Type and Position in Image Scene
* **Vedoucí:** Prof. Ing., Dipl.-Ing. Martin Drahanský, Ph.D., UITS FIT VUT
* **Oponent:** Ing. Tomáš Goldmann, UITS FIT VUT
* **Student:** Kolcún Róbert

## Zadání:
1. Prostudujte literaturu týkající se výskytu objektů v obrazu a seznamte se s algoritmy pro jejich detekci a rozpoznávání.
2. Navrhněte algoritmický postup pro stanovení typu zbraně (krátká, dlouhá, vč. příp. dalšího jemnějšího dělení) a jejího natočení ve scéně.
3. Postup navržený v předchozím bodu implementujte. Proveďte otestování Vašeho řešení.
4. Shrňte dosažené výsledky a diskutujte možnosti budoucího vývoje.

## Literatura:
* Olmos R., Tabik S., Herrera S. Automatic Handgun Detection Alarm in Videos Using Deep Learning. Neurocomputing, 2017, DOI https://doi.org/10.1016/j.neucom.2017.05.012
* Lai J., Maples S. Developing a Real-Time Gun Detection Classifier. Dostupný on-line: http://cs231n.stanford.edu/reports/2017/pdfs/716.pdf

## Obsah priečinka
* dataset
    * priečinok angle\_detection obsahuje 2 podpriečinky.
        * training_data - obrazky na ktorých prebiehalo trénovanie modelov.
        * test_data - zvyšné obrázky ktoré neboli použité ani na trénovanie ani na testovanie.
    * priečinok classification obsahuje rovnako 2 podpriečinky s rovnakým významom ako priečinok angle\_detection.
* doc - priečinok obsahujúci všetky tex-ovské zdrojové súbory, bibliografiu a obrázky použité v bakalárskej práci.
* models - priečinok obsahuje 4 podpriečinky v ktorých sú natrénovane modely pre určenie typu a náklonu zbrane.
    * cnnc - model pre klasifikáciu typu zbrane
    * cnna\_pitch - model pre určenie náklonu zbrane v ose pitch
    * cnna\_roll - model pre určenie náklonu zbrane v ose roll
    * cnna\_yaw - model pre určenie náklonu zbrane v ose yaw
* src - zdrojové súbory implementácie
    * dataset\_processing - obsahuje pomocné scripty použité pri stahovaní a generovaní vstupných dát pre trénovanie modelov
    * models - všetky zdrojové súbory použité pre trénovanie, testovanie a tvorbu modelov, ich podrobná  implementácia je opísaná v kapitole 4.Implementácia v bakalárskej práci
* Pipfile a Pipfile.lock - konfiguračné súbory pre nástroj pipenv, ktorý vytvára virtuálne prostredie pre programovací jazyk Python

## Postup spustenia programu
1. Pre spustenie programu je potrebné mať nainštalovaný nástroj pipenv a python3.6
2. Vytvorenie virtuálneho prostredia a inštalácia všetkých potrebných balíčkov
```bash
$ pipenv shell
(virtualenv) $ pipenv install
```
3. Doinštalovanie tensorflow knižnice
```bash
(virtualenv) $ pip install tensorflow
```
4. Spustenie programu predikcie
```bash
(virtualenv) $ python src/models/predict.py --class models/cnnc/ --anglep models/cnna_pitch/ --angler models/cnna_roll/ --angley models/cnna_yaw/ --image path_to_image
```
