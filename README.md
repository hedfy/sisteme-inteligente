# sisteme-inteligente
# Student Performance Prediction using Machine Learning

## Descriere
Acest proiect are ca scop prezicerea performanței elevilor folosind algoritmi de Machine Learning.

## Sursa Datelor
Dataset: Students Performance in Exams

Link:
https://www.kaggle.com/datasets/spscientist/students-performance-in-exams


## Obiectiv
Scopul proiectului este de a analiza factorii care influențează performanța studenților și de a construi modele capabile să prezică rezultatele la examene.

## Tipul problemei
Problemă de regresie.

## Tehnologii utilizate
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Structura proiectului

- data/ -> conține setul de date
- notebooks/ -> analiză exploratorie și experimente
- src/ -> codul sursă
- docs/ -> documentația proiectului

## Algoritmi planificați
- Linear Regression
- Decision Tree
- Random Forest

## Autor
[Alex Herman]

# DOCUMENTAȚIE PROIECT

## Analiza și Predicția Performanței Elevilor

# 📊 Analiza și Predicția Performanței Școlare a Elevilor

## 1. Introducere
Proiectul de față reprezintă o aplicație complexă de **Data Science** și **Machine Learning**, dezvoltată în mediul interactiv *Jupyter Notebook*. Scopul principal al proiectului este analiza performanței școlare a elevilor și construirea unui model predictiv capabil să estimeze scorul la matematică pe baza unor factori demografici și educaționali.

Setul de date utilizat – `StudentsPerformance.csv` – conține înregistrări despre elevi din diverse medii socio-economice și culturale. Fiecare înregistrare include informații despre:
* Genul elevului
* Grupul etnic de proveniență
* Nivelul de educație al părinților
* Tipul de program alimentar la care participă elevul
* Absolvirea unui curs de pregătire pentru testare
* Scorurile obținute la trei discipline: **matematică**, **citire** și **scriere**.

Prin utilizarea unor biblioteci moderne din ecosistemul Python – `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `Scikit-Learn` și `SHAP` – proiectul urmărește un flux complet de lucru în Data Science, de la colectarea și preprocesarea datelor, până la antrenarea și explicarea unui model predictiv.

### 1.1 Motivație și relevanță
Analiza performanței academice reprezintă un domeniu de mare interes atât în cercetarea educațională, cât și în aplicațiile de inteligență artificială. Identificarea timpurie a factorilor care influențează rezultatele școlare permite intervenții educaționale mai eficiente și personalizate.

Prin construirea unui model de Machine Learning, se poate anticipa performanța unui elev înainte ca aceasta să fie evaluată formal, oferind profesorilor și instituțiilor de învățământ un instrument valoros de decizie.

### 1.2 Obiective
* **Explorarea și înțelegerea** structurii setului de date privind performanța elevilor.
* **Identificarea corelațiilor** și relațiilor dintre variabilele educaționale.
* **Preprocesarea și transformarea** datelor pentru compatibilitatea cu algoritmii de ML.
* **Antrenarea și compararea** mai multor modele de regresie.
* **Optimizarea modelului** selectat prin tehnici de tuning al hiperparametrilor.
* **Explicarea predicțiilor** folosind biblioteca SHAP pentru transparență și interpretabilitate.

* ## 2. Tehnologii și Biblioteci Utilizate

Proiectul utilizează un ecosistem complet de instrumente Python dedicate analizei de date și Machine Learning. Tabelul următor prezintă principalele tehnologii folosite, versiunile recomandate și scopul fiecăreia în cadrul proiectului:

| Tehnologie | Versiune | Scop principal |
| :--- | :--- | :--- |
| **Python** | 3.10+ | Limbaj de programare principal |
| **Pandas** | 2.x | Manipulare și procesare date |
| **NumPy** | 1.x | Calcule numerice și matriciale |
| **Matplotlib** | 3.x | Vizualizare grafică 2D |
| **Seaborn** | 0.12+ | Grafice statistice avansate |
| **Scikit-Learn** | 1.x | Algoritmi de Machine Learning |
| **SHAP** | 0.41+ | Explicabilitate modele ML |
| **Jupyter Notebook** | 6.x+ | Mediu de dezvoltare interactiv |

### 2.1 Mediul de dezvoltare
**Jupyter Notebook** a fost ales ca mediu de lucru datorită interactivității sale și capacității de a combina cod Python, rezultate vizuale și text explicativ în același document. Această abordare facilitează documentarea pas-cu-pas a procesului de analiză.

### 2.2 Justificarea alegerii bibliotecilor
* **Pandas** și **NumPy** formează coloana vertebrală a manipulării datelor, oferind structuri de date eficiente (`DataFrames`) și operații matematice optimizate.
* **Matplotlib** și **Seaborn** completează stack-ul cu vizualizări clare, intuitive și estetice pentru analiza exploratorie.
* **Scikit-Learn** asigură un API unificat și robust pentru preprocesare, antrenare și evaluarea modelelor de regresie prin intermediul pipeline-urilor.
* **SHAP** adaugă transparența critică necesară interpretării rezultatelor, transformând un model complex într-unul ușor de înțeles (XAI - *Explainable AI*).

* ## 3. Setul de Date

### 3.1 Descrierea setului de date
Setul de date `StudentsPerformance.csv` este un dataset public ce conține informații despre **1.000 de elevi** și performanțele acestora la testele standardizate. Acesta este frecvent utilizat ca exemplu de referință în proiectele și tutorialele de Data Science datorită echilibrului său, varietății variabilelor și structurii sale curate.

### 3.2 Structura și variabilele

Setul de date îmbină factori demografici, socio-economici și indicatori de pregătire școlară, structurați astfel:

| Variabilă | Tip Date | Descriere / Valori posibile |
| :--- | :--- | :--- |
| `gender` | Categoric | Genul elevului (`male` / `female`) |
| `race/ethnicity` | Categoric | Grupul etnic anonimizat (`group A`, `group B`, `group C`, `group D`, `group E`) |
| `parental level of education` | Categoric | Nivelul de educație al părinților (`some high school`, `high school`, `some college`, `associate's degree`, `bachelor's degree`, `master's degree`) |
| `lunch` | Categoric | Tipul de prânz (indicator socio-economic): (`standard` / `free/reduced`) |
| `test preparation course` | Categoric | Participarea la un curs pregătitor (`none` / `completed`) |
| **`math score`** | **Numeric** | **Scorul la matematică (Variabila țintă / Target, valori: 0–100)** |
| `reading score` | Numeric | Scorul la citire (valori: 0–100) |
| `writing score` | Numeric | Scorul la scriere (valori: 0–100) |

### 3.3 Caracteristici statistice

> 📊 **Sumar Statistic Rapid:**
> * **Distribuție:** Scorurile la cele trei discipline (matematică, citire, scriere) urmează o distribuție aproximativ normală (Gaussiană).
> * **Medie:** Performanța medie a elevilor se situează stabil în intervalul **65–68 de puncte** pentru toate cele trei materii.
> * **Variabilitate:** Deviația standard se poziționează în jurul valorii de **15 puncte**, fapt ce indică o împrăștiere moderată a notelor în rândul populației analizate.
> * **Extremis:** Valorile acoperă întreaga scală disponibilă, de la **0 la 100**, demonstrând o diversitate ridicată și existența unor profile atipice în rândul elevilor (de la performanțe excepționale până la cazuri severe de abandon/eșec școlar).
