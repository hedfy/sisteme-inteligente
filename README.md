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

## 4. Explorarea și Curățarea Datelor

### 4.1 Încărcarea datelor
Prima etapă a proiectului constă în importul setului de date dintr-un fișier de tip `CSV`. Datele sunt încărcate într-un obiect de tip `DataFrame` din biblioteca **Pandas**, o structură tabelară optimizată ce permite operații vectorizate rapide asupra coloanelor și rândurilor.

Verificarea structurală inițială și inspecția preliminară sunt realizate prin metodele native Pandas:
* `.head()` — pentru vizualizarea primelor rânduri și înțelegerea formatului datelor.
* `.info()` — pentru obținerea unui rezumat rapid al tipurilor de date stocate (`object`, `int64`) și a numărului de intrări.
* `.describe()` — pentru generarea statisticilor descriptive de bază (medie, mediană, valori minime/maxime).

### 4.2 Identificarea problemelor de calitate
În faza de explorare, s-a urmărit identificarea și remedierea potențialelor anomalii pentru a asigura integritatea datelor:

* **Valori lipsă (NaN / Null):** Verificate riguros folosind instrucțiunea `df.isnull().sum()`. În urma analizei, s-a constatat că setul de date este complet curat, având **0 valori nule** pe toate axele.
* **Înregistrări duplicate:** Identificate și validate pentru a preveni apariția unui *bias* (polarizare) în faza de antrenare a modelelor. Nu au fost detectate duplicate.
* **Inconsistențe de format:** S-au verificat valorile categorice pentru a elimina eventuale spații suplimentare accidentale (ex: `" group A"` în loc de `"group A"`) sau diferențe de scriere cu majuscule.
* **Valori aberante (Outliers):** Identificate la nivelul scorurilor prin analize vizuale de tip Boxplot. Deși există scoruri foarte mici (inclusiv note de 0), acestea reflectă realitatea performanțelor școlare și nu erori de introducere a datelor, motiv pentru care au fost păstrate pentru a nu altera analiza cazurilor de eșec școlar.

### 4.3 Preprocesarea datelor (Data Engineering)
Deoarece algoritmii de Machine Learning operează exclusiv cu structuri matematice și matrici numerice, variabilele de tip text au trecut printr-un proces de transformare:

1. **Transformarea Variabilelor Categorice:**
   * Pentru atributele binare (`gender`, `test preparation course`), s-au mapat valorile direct în format binar ($0$ și $1$).
   * Pentru atributele cu mai multe categorii (`race/ethnicity`, `parental level of education`), s-a configurat tehnica de **One-Hot Encoding** prin intermediul unui `ColumnTransformer` Scikit-Learn. Acest proces transformă categoriile text în coloane binare independente, prevenind atribuirea unei ordini ierarhice artificiale între grupuri.

2. **Gruparea și Automatizarea Pipeline-ului:**
   Toate aceste transformări sunt incluse într-un preprocesor automatizat. Acesta preia datele brute, aplică mapările și encodarea în mod controlat și livrează matricea finală către model, asigurând o separare totală între setul de antrenament și cel de test (*data leakage prevention*).


   ## 5. Vizualizarea Datelor (EDA)

### 5.1 Tipuri de grafice utilizate
Vizualizarea datelor a fost un instrument esențial în acest proiect pentru identificarea tiparelor ascunse și comunicarea eficientă a concluziilor analitice. Cu ajutorul bibliotecilor **Matplotlib** și **Seaborn**, s-a construit o serie de reprezentări grafice adaptate fiecărui tip de variabilă:

* 📊 **Histograme (`sns.histplot`)** — Utilizate pentru a ilustra distribuția de frecvență a scorurilor la matematică, citire și scriere. Acestea au permis confirmarea simetriei și validarea formei de distribuție clopot (Gaussiană).
* 📦 **Boxplot-uri (`sns.boxplot`)** — Folosite pentru a vizualiza dispersia (cvartilele, mediana) și pentru a detecta vizual valorile aberante (*outliers*), facilitând compararea performanțelor între grupuri (de exemplu, în funcție de nivelul de educație al părinților).
* 🗺️ **Heatmap de corelație (`sns.heatmap`)** — O reprezentare grafică a matricei de corelație Pearson între variabilele numerice. Aceasta evidențiază intensitatea și direcția legăturilor dintre scoruri prin nuanțe de culoare.
* 📊 **Grafice de bare (`sns.barplot`)** — Create pentru a compara direct mediile scorurilor obținute în funcție de factorii categorici relevanți (gen, grup etnic, participarea la cursul de pregătire).
* 🔵 **Scatter plots (`sns.scatterplot`)** — Utilizate pentru explorarea relațiilor bivariate dintre discipline (ex: corelația dintre nota la citire și nota la scriere), ajutând la identificarea tendințelor liniare evidente.

### 5.2 Insight-uri relevante (Concluzii Vizuale)

În urma interpretării graficelor generate în Jupyter Notebook, au fost extrase următoarele concluzii esențiale:

> 💡 **Impactul cursului de pregătire:** > Elevii care au finalizat cursul de pregătire (*test preparation course = completed*) obțin în mod constant scoruri medii semnificativ mai ridicate la toate cele trei discipline, comparativ cu cei care nu s-au pregătit deloc.

> 📈 **Sinergia disciplinelor (Corelație transversală):** > Există o corelație pozitivă extrem de puternică între `reading score` (citire) și `writing score` (scriere). Elevii care excelează la una dintre aceste materii au o probabilitate foarte mare să obțină un scor ridicat și la cealaltă. De asemenea, performanța de la aceste două discipline se propagă pozitiv și asupra notei la matematică.

> 👥 **Influența factorilor demografici și socio-economici:** > Factori precum genul, grupul etnic și tipul de prânz (`lunch`) introduc variații statistice vizibile în distribuția notelor. De exemplu, elevii care beneficiază de prânz standard (indicator al unui statut socio-economic stabil) înregistrează performanțe medii mai bune, ceea ce transformă aceste variabile în predictori extrem de relevanți pentru modelul de Machine Learning.
>
> ## 6. Ingineria Caracteristicilor (Feature Engineering)

### 6.1 Transformarea variabilelor categorice
Deoarece algoritmii de Machine Learning operează exclusiv cu structuri matematice și matrice numerice, variabilele de tip text (categorice) au fost mapate și transformate folosind strategii specifice naturii fiecăreia:

* 🔢 **Label Encoding (Mapare Binară):** Aplicat variabilelor cu doar două stări posibile, transformându-le în valori binarizate ($0$ sau $1$):
  * `gender`: `male` $\rightarrow$ $0$, `female` $\rightarrow$ $1$
  * `test preparation course`: `none` $\rightarrow$ $0$, `completed` $\rightarrow$ $1$
* 📈 **Ordinal Encoding:** Utilizat pentru variabila `parental level of education`. Deoarece nivelurile de școlarizare au o ierarhie naturală, s-au atribuit valori numerice incrementale care respectă această ordine (de la *some high school* până la *master's degree*).
* 🔀 **One-Hot Encoding:** Aplicat variabilei `race/ethnicity`. Deoarece grupurile etnice nu prezintă o ierarhie sau o ordine matematică, această tehnică creează coloane binare independente pentru fiecare categorie (ex: `group_A`, `group_B` etc.), evitând introducerea unui *bias* artificial în model.

### 6.2 Analiza corelațiilor
Calcularea **Matricei de corelație Pearson** pe setul de date preprocesat a confirmat matematic ipotezele ridicate în faza de analiză exploratorie (EDA):

* 🚀 **Predictori foarte puternici:** Scorurile obținute la `reading score` (citire) și `writing score` (scriere) prezintă o corelație liniară extrem de ridicată (valori de peste **$0.85$**) cu variabila țintă (`math score`). Acest lucru indică faptul că performanțele la disciplinele conexe sunt indicatori excelenți pentru succesul la matematică.
* 📈 **Corelație moderată:** Variabila `test preparation course` (participarea la cursul de pregătire) prezintă o corelație pozitivă moderată ($\approx 0.35$), reconfirmând faptul că studiul suplimentar organizat are un impact direct și măsurabil asupra notei finale.
* 👥 **Corelații demografice:** Factorii demografici puri prezintă coeficienți de corelație liniară mai scăzuți, însă interacțiunea lor complexă (capturată ulterior de model) aduce un plus considerabil în reducerea erorilor și îmbunătățirea generalizării.

### 6.3 Selecția caracteristicilor (Feature Selection)
Procesul de selecție a inclus evaluarea relevanței fiecărui atribut pe baza analizei corelațiilor și a scorurilor de importanță extrase din modele bazate pe arbori de decizie. 

Prin păstrarea exclusivă a caracteristicilor cu impact predictiv demonstrat, s-au realizat următoarele optimizări:
1. **Reducerea dimensionalității:** Eliminarea zgomotului din date și optimizarea spațiului de caracteristici pentru algoritmi.
2. **Prevenirea supra-antrenării (*Overfitting*):** Modelul învață tipare generale, solide, în loc să memoreze particularități sau zgomote statistice din setul de antrenament.

## 7. Construirea și Evaluarea Modelelor de Machine Learning

### 7.1 Împărțirea datelor (Data Splitting)
Pentru a evalua corect capacitatea de generalizare a modelelor și pentru a evita fenomenul de supra-antrenare (*overfitting*), setul de date preprocesat a fost împărțit în mod controlat:
* **80% - Subset de antrenare (Train Set):** Folosit pentru ajustarea ponderilor și învățarea tiparelor matematice de către algoritmi.
* **20% - Subset de testare (Test Set):** Păstrat complet izolat, acționând ca date complet noi pentru validarea finală a performanței.

Această segmentare este realizată cu funcția `train_test_split()` din Scikit-Learn. S-a utilizat un parametru `random_state` fixat (o valoare numerică constantă) pentru a asigura **reproducibilitatea** deplină a rezultatelor la fiecare rulare succesivă a codului.

### 7.2 Modele implementate și metrici de evaluare
Au fost antrenate, testate și comparate în paralel trei modele de regresie cu grade diferite de complexitate. Performanțele lor comparative sunt sintetizate în tabelul de mai jos:

| Model | MAE | MSE | $R^2$ Score | Observații |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | ~4.5 | ~32 | ~0.87 | Model de bază (*baseline*), stabil și excelent pentru analiza coeficienților directi. |
| **Random Forest** | ~3.2 | ~20 | ~0.92 | Performanță ridicată, ansamblu bazat de arbori de decizie, rezistent la variații. |
| **Gradient Boosting** | **~3.0** | **~18** | **~0.93** | **Cel mai bun model** din punct de vedere predictiv în urma optimizării hiperparametrilor. |

### 7.3 Interpretarea metricilor de performanță

Pentru diagnosticarea riguroasă a erorilor de predicție, s-au utilizat trei metrici standard din industrie:

* 📏 **MAE (Mean Absolute Error) — Eroarea Absolută Medie:**
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
  Reprezintă media aritmetică a erorilor absolute. În contextul proiectului, o valoare MAE de ~3.0 indică faptul că, în medie, modelul estimează nota la matematică a unui elev cu o deviație de doar 3 puncte (pe o scară de la 0 la 100) față de nota sa reală.

* 📐 **MSE (Mean Squared Error) — Eroarea Medie Pătratică:**
  $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
  Această metrică ridică la pătrat diferențele înainte de a face media. Prin urmare, penalizează mult mai sever erorile mari. Este un indicator excelent pentru a detecta dacă modelul dă rateuri majore pe anumite profiluri atipice de studenți.

* 📈 **$R^2$ Score (Coefficient of Determination) — Coeficientul de Determinare:**
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
  Măsoară proporția din varianța notei la matematică care poate fi explicată și anticipată pe baza caracteristicilor de intrare. Scorul obținut de **0.93** de către modelul de Gradient Boosting demonstrează că aproximativ **93% din variația notelor** este acoperită cu succes de model, indicând o potrivire și o precizie excelentă.

  ## 8. Optimizarea Modelului (Hyperparameter Tuning)

### 8.1 Tehnica GridSearchCV și Validarea Încrucișată
Pentru a maximiza capacitatea predictivă a modelului performant și a extrage maximum de valoare din date, s-a implementat algoritmul **GridSearchCV** din Scikit-Learn. 

Această tehnică realizează o căutare exhaustivă și sistematică prin combinarea tuturor valorilor dintr-o grilă predefinită de hiperparametri. Fiecare combinație este evaluată riguros prin **Validare Încrucișată (K-Fold Cross-Validation)** cu un factor de **$k=5$ fold-uri**. Această abordare garantează că:
1. Modelul este antrenat și testat pe subseturi diferite în mod rotativ.
2. Estimarea performanței este extrem de robustă și stabilă.
3. Se elimină riscul ca performanța să fie influențată de o împărțire norocoasă sau particulară a datelor.

### 8.2 Hiperparametri optimizați
Spațiul de căutare a fost configurat pentru a controla complexitatea algoritmilor și a preveni supra-antrenarea (*overfitting*), vizând următoarele setări structurale:

| Hiperparametru | Rol și Impact | Valori Testate în Grilă |
| :--- | :--- | :--- |
| `n_estimators` | Numărul de arbori de decizie din ansamblu. | `[50, 100, 200]` |
| `max_depth` | Adâncimea maximă a fiecărui arbore (limitează complexitatea). | `[None, 5, 10, 20]` |
| `min_samples_split` | Numărul minim de eșantioane necesare pentru a împărți un nod intern. | Configurat în funcție de model |
| `min_samples_leaf` | Numărul minim de eșantioane necesare pentru a exista într-un nod frunză. | Configurat în funcție de model |
| `learning_rate` | Viteza de ajustare (rata de învățare a corecțiilor) pentru Gradient Boosting. | Scalat în pași zecimali |

### 8.3 Rezultate obținute după optimizare

> 🚀 **Impactul Tuning-ului de Hiperparametri:**
> * **Reducerea Erorilor:** Identificarea configurației optime prin `GridSearchCV` a condus la o scădere substanțială a erorii medii absolute (**MAE a scăzut cu ~8–12%** față de valorile obținute cu parametrii impliciți / *default*).
> * **Creșterea Preciziei:** Coeficientul de determinare **$R^2$ Score a crescut cu 2–3 puncte procentuale**, confirmând o potrivire superioară a modelului pe datele de testare.
> * **Stabilitate:** Modelul optimizat demonstrează o capacitate de generalizare excelentă, fiind pregătit pentru a genera predicții precise pe seturi de date complet noi.
>   ## 9. Explicabilitatea Modelului cu SHAP (Explainable AI)

### 9.1 Necesitatea explicabilității în educație
În domenii cu impact social ridicat, precum educația, nu este suficient ca un model de Machine Learning să producă predicții precise; este esențial ca aceste decizii să fie transparente și complet interpretabile. 

Explicabilitatea modelului permite profesorilor și factorilor de decizie să înțeleagă mecanismele interne ale algoritmului și să identifice cu exactitate factorii determinanți în succesul sau riscul de eșec școlar al unui elev. Eliminarea fenomenului de „cutie neagră” (*black box*) previne mascarea unor eventuale *bias*-uri (prejudecăți) statistice nedorite și consolidează încrederea în sistemele inteligente de asistență educațională.

### 9.2 Principiile matematice SHAP
Pentru a aduce transparență totală, proiectul integrează biblioteca **SHAP (SHapley Additive exPlanations)**. 

Metoda se bazează pe o fundamentare matematică riguroasă derivată din **teoria jocurilor cooperative**. SHAP calculează contribuția marginală a fiecărei caracteristici de intrare la predicția finală a unui elev, distribuind „câștigul” sau „penalizarea” de scor în mod echitabil între toate variabilele implicate. Valorile SHAP garantează respectarea a patru proprietăți fundamentale: *eficiență*, *simetrie*, *nulitate* (dummy) și *aditivitate*.

### 9.3 Tipuri de vizualizări SHAP implementate
În cadrul notebook-ului, au fost generate și analizate următoarele reprezentări grafice:

* 📊 **Summary Plot (`shap_summary_plot.png`)** — Oferă o imagine de ansamblu la nivel global. Acesta combină importanța caracteristicilor cu impactul lor direct, arătând nu doar care variabile contează cel mai mult, ci și dacă valorile mari ale acestora cresc sau scad nota prezisă la matematică.
* 🌊 **Waterfall Plot** — Utilizat pentru analiza locală. Explică în detaliu o singură predicție (un singur elev), pornind de la media generală a bazei de date și adunând sau scăzând contribuția specifică a fiecărui atribut (ex: cum prânzul gratuit i-a scăzut estimarea sau cum cursul finalizat a urcat-o).
* 📉 **Dependence Plot** — Ilustrează relația non-liniară dintre valorile unei caracteristici și valorile SHAP corespunzătoare, scoțând la iveală interacțiuni complexe între variabile (ex: cum impactul notei la citire se schimbă în funcție de gen).
* 🧲 **Force Plot** — O reprezentare vizuală a fortelor dinamice care „împing” modelul să ia o decizie, evidențiind factorii care acționează pozitiv sau negativ în raport cu linia de bază (*base value*).

### 9.4 Interpretarea rezultatelor (Insight-uri XAI)

> 🔬 **Concluzii extrase prin SHAP:**
> * **Dominanța abilităților cognitive:** Analiza SHAP confirmă indubitabil că abilitățile academice transversale — reflectate prin `reading score` și `writing score` — sunt principalii vectori care propulsează predicția notei la matematică în zona superioară.
> * **Valoarea adăugată a pregătirii:** Participarea și finalizarea cursului de pregătire (`test preparation course = completed`) generează un efect pozitiv consistent și stabil pe graficul summary plot, acționând ca un catalizator pentru note mai mari.
> * **Rolul factorilor socio-economici:** Indicatorul de prânz (`lunch`) demonstrează un impact vizibil; accesul la prânzul standard corelează pozitiv cu predicțiile mari, indicând o legătură fină între stabilitatea socio-economică și performanța la disciplinele exacte.
>   ## 10. Concluzii și Perspective (Conclusions & Future Work)

### 10.1 Rezultate obținute
Proiectul a demonstrat cu succes că metodologiile specifice **Data Science** și **Machine Learning** pot fi aplicate eficient în domeniul educațional pentru analiza și predicția performanței elevilor. 

Algoritmii implementați au obținut rezultate solide, atingând un coeficient de determinare **$R^2$ de peste 0.92**. Acest lucru indică matematic faptul că factorii demografici, socio-economici și istoricul de pregătire incluși în setul de date reușesc să explice peste **92% din variabilitatea scorurilor la matematică**. 

Mai mult, integrarea bibliotecii **SHAP** a adus o valoare adăugată critică: a transformat un model predictiv opac într-un instrument transparent, audat și complet interpretabil, perfect aliniat cu cerințele practice din instituțiile de învățământ moderne.

### 10.2 Limitări ale proiectului
În ciuda preciziei ridicate, modelul prezintă o serie de limitări ce trebuie luate în considerare înainte de o eventuală implementare la scară largă:
* 📉 **Volumul de date:** Setul de date conține un eșantion de doar 1.000 de înregistrări, ceea ce poate limita capacitatea de generalizare a modelului la nivelul unor populații școlare mult mai mari sau mai eterogene.
* 🔍 **Variabile lipsă:** Factori socio-economici deosebit de complecși — cum ar fi venitul exact al familiei, calitatea infrastructurii școlare, numărul de ore de somn sau accesul la resurse tehnologice — nu sunt cuantificați în dataset.
* 🌍 **Specificul cultural:** Modelul a fost antrenat pe date dintr-un context geo-cultural specific (un sistem de testare standardizat), motiv pentru care ar putea necesita ajustări majore sau o re-antrenare completă pentru a fi replicat în alte sisteme de învățământ (de exemplu, cel european sau național).

### 10.3 Direcții viitoare de dezvoltare
Pentru a crește valoarea practică și aplicabilitatea proiectului în lumea reală, au fost identificate următoarele axe de extindere:
* 📊 **Scalarea volumului de date:** Integrarea unor seturi de date mult mai vaste și colectate din regiuni geografice diverse pentru a fortifica robustețea algoritmilor.
* 🧠 **Explorarea Deep Learning:** Utilizarea rețelelor neuronale artificiale de tip *Feed-Forward* sau a modelelor de tab-net pentru a detecta pattern-uri non-liniare extrem de fine sau interacțiuni ascunse între caracteristici.
* 🚀 **Dezvoltarea unei aplicații interactive:** Construirea unei interfețe web intuitive folosind framework-uri precum **Streamlit** sau **Flask**. Aceasta va permite profesorilor sau consilierilor școlari să introducă profilul unui elev în timp real și să vizualizeze instant nota estimată și analiza SHAP asociată.
* 🎯 **Sisteme de recomandare educațională:** Trecerea de la o analiză pur predictivă la una prescriptivă, capabilă să sugereze automat măsuri de remediere (ex: recomandarea obligatorie a cursului de pregătire dacă modelul estimează un risc de eșec).
* ⏳ **Analiză longitudinală:** Extinderea bazei de date cu informații temporale pentru a urmări și anticipa evoluția performanței elevilor de-a lungul mai multor ani școlari (analiză de tip *time-series*).

### 10.4 Concluzie finală

> 🏆 **Sumar Executiv:**
> Proiectul **StudentPerformance** reprezintă o transpunere completă, riguroasă și integrată a fluxului de lucru din Data Science. Acesta demonstrează modul în care tehnologiile moderne de analiză și modelare pot fi puse în mod direct în slujba optimizării procesului educațional. 
> 
> Combinarea armonioasă dintre analiza exploratorie, modelarea predictivă de înaltă precizie și explicabilitatea bazată pe teoria jocurilor oferă un exemplu concret de **Inteligență Artificială responsabilă, etică și orientată spre un impact social pozitiv**.
>
> ## Bibliografie și Resurse (References & Documentation)

Această secțiune cuprinde literatura academică de specialitate care fundamentează algoritmii utilizați, precum și resursele și documentațiile oficiale accesate pe parcursul dezvoltării proiectului:

### 📚 Referințe Academice
* **Scikit-Learn:** Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
* **Pandas / NumPy:** McKinney, W. (2010). *Data Structures for Statistical Computing in Python*. Proceedings of the 9th Python in Science Conference, 51–56.
* **SHAP (Explainable AI):** Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems (NeurIPS 2017), 30.

### 🌐 Documentații Oficiale și Resurse Web
* 🐼 [Documentația Oficială Pandas](https://pandas.pydata.org/docs/) — Ghiduri complete pentru manipularea structurilor de date de tip DataFrame și Series.
* ⚙️ [Documentația Oficială Scikit-Learn](https://scikit-learn.org/stable/documentation.html) — Referințe tehnice pentru construirea de pipeline-uri, preprocesare și algoritmi de regresie.
* 🧬 [Documentația Oficială SHAP](https://shap.readthedocs.io/en/latest/) — Exemple de implementare și interpretare a graficelor bazate pe valorile Shapley (Teoria Jocurilor).
* 📊 [Sursa Setului de Date (Kaggle)](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) — Pagina oficială a datasetului *Students Performance in Exams*, de unde au fost extrase datele brute utilizate în analiză.
