Как борются с фродом в такси

В настоящее время в моделях выявления фрода в такси в основном используются данные о регистрации водителей и пассажиров, идентификаторы смартфона и SIM, данные спутникового и мобильного геопозиционирования, платежные данные пользователей.

Наиболее частым является Rule-based подход, который заключается в составлении совокупности правил для выявления отклонений в данных, соответствующих шаблонам мошеннических действий. Проверка правил производится автоматически, результат проверки отправляется аналитику для принятия окончательного решения о наличии фрода. Rule-based подход используется в России как крупными агрегаторами, такими как Ситимобил [1], Яндекс Такси [2], так и мелкими [3, 4], а также, например, крупным американским Lyft [5].

Rule-based подход имеет следующие недостатки: сложность в составлении универсальных правил, необходимость применения domain-specific языка описания, ограниченность наборов правил для выявления новых случаев, ручная проверка результатов. Решение перечисленных недостатков, а также повышение скорости обработки данных и качества принятия решения лежит в гиперплоскости Big Data и Machine Learning.

Так, известно, что мировой лидер по такси-перевозкам, компания Uber для выявления фрода использует данные сообщества миллионов пользователей, как водителей, которых насчитывается более 5 миллионов, так и пассажиров, которых порядка 100 миллионов. Система антифрода в Uber построена с применением графовой сверточной сети (Relational Graph Convolutional Networks) [6].

Российский IT-гигант Яндекс имеет большую экспертизу в части антифрода для своих проектов Директ и Деньги. То, что Яндекс активно и давно развивает технологии машинного обучения и внедряет их в продакшен позволяет предположить о наличии ML-разработок по антифроду в Такси, в том числе и на основе данных пользователей всей его экосистемы продуктов.

Сведения по антифроду в крупных агрегаторах Европы (Bolt, Gett), Китая (Didi), Индии (Ola Cabs) и стран Юго-Восточной Азии (Grab, Go-Jek) в свободном доступе отсутствуют. Но опубликованные на их сайтах вакансии говорят о том, что эти компании активно набирают свои ML-команды и развивают DS-направление для поиска андтифрода.

Задачи машинного обучения в антифроде в такси две: первая - классификация и кластеризация водителей и пассажиров на основе категориальных и числовых признаков; вторая - поиск аномалий во временных рядах с историями поездок.

Задачи классификации и кластеризации пользователей это SOTA, для их решения разработан широкий спектр как моделей традиционного машинного обучения, так и глубокого обучения. В рамках данного обзора такие задачи мы не рассматриваем.

Обратимся к задаче поиска аномалий во временных рядах с историями поездок. Данные для решения этой задачи состоят из треков показаний датчика положения (GPS) и датчика скорости (акселерометр), а также могут включать или не включать сведения из профиля водителя и пассажира.

Рассматриваем случай, когда поиск аномалий в историях поездок осуществляется только по данным GPS и акселерометра. Такая задача относится к категории задач Trajectory Outlier Detection.

Различные научные статьи в этом направлении регулярно выходят с 2008 года. В 2020 году вышел обзор методов машинного обучения для решения задачи поиска аномалий в траекториях [7], на этот обзор мы и будем ориентироваться.


Для глубокого погружения в тематику поиска аномалий во временных рядах, содержащих данные геопозиционирования, стоит отметить следующие работы:
- Gupta et al. - Поиск аномалий во временных рядах [8], 2014;
- Schubert et al. - Формализация задач поиска аномалий во временных рядах на основе анализа соседей [9], 2014;
- Zheng et al. - Urban Computing: концепции, методологии и приложения [10], 2014;
- Zheng - Обзор data-mining задач для анализа траекторий [11], 2015;
- Feng and Zhu - Обзор data-mining задач для анализа траекторий [12], 2016;
- Belhadi et al. - Сравнительный анализ алгоритмов поиска аномалий в траекториях [13], 2019;
- Djenouri et al. - Обзор алгоритмов поиска аномалий городского трафика [14], 2019;
- Alsrehin et al. - Обзор техник машинного обучения и анализа данных в области перевозок и дорожного трафика [15], 2019;
- Chalapathy and Chawla - Обзор алгоритмов глубокого обучения в задачах поиска аномалий [16], 2019.


Известные алгоритмы поиска аномалий во временных рядах:
- TRAOD (TRAjectory Outlier Detection) - алгоритм разбиения на интервалы (t-partitions), и поиска аномалий с помощью сравнения перпендикулярной, параллельной и угловой составляющих расстояния;
- SVM Trajectory Outlier Detector - применение метода опорных векторов для группировки траекторий по схожим признакам;
- Clustering Trajectory Outlier Detection (CTOD) - применение метода DBSCAN для кластеризации интервалов траекторий по расстояниям между минимальными описывающими прямоугольниками;
- TOP-EYE - анализ векторов разложений интервалов траектории по сетке направлений;
- Lifted Trajectories and kNN - уменьшение размерности путем сжатия траекторий;
- iBAT (isolation-Based Anomalous Trajectory) - применение изолирующего леса;
- RTOD (Relative distance-based Trajectory Outlier Detection) - расчет метрики Хаусдорда на множестве точек траектории;
- RPAT (Road segment Partitioning towards Anomalous Trajectory detection) - разложение траекторий по базису дорожной сети;
- iBOAT (isolation-Based Online Anomalous Trajectory) - построение пространственной сетки и применение процедуры изоляции аномалий;
- SHNN-CAD (Sequential Hausdorff Nearest Neighbours Conformal Anomaly Detector) - применение модели Conformal Prediction для подсчета доверительных интервалов по метрике Хаусдорфа;
- PN-Outlier (Point-Neighbor-based trajectory Outlier) - подсчет числа аномальных соседних точек на интервалах траекторий;
- Detect - маркировка дорожной сети метками о частоте движения на отдельных участках дорог;
- BT-miner - поиск аномалий по данным, дополненным данными мобильной сети;
- TPRO (Time-dependent Popular Routes-based trajectory Outlier detection) - построение наиболее частых маршрутов;
- ACE (Anomaly Clustering Ensemble) - ансамбль моделей кластеризации с разным разрешением;
- iBDD (isolation-Based Disorientation Detection) - подсчет исторических данных для выявления совпадающих участков траекторий;
- MANTRA (Maximal ANomalous sub-TRAjectories) - подсчет временных интервалов между участками траекторий;
- MT-MAD (Maritime Trajectory Modeling and Anomaly Detection) - анализ частных, последовательных и поведенческих отклонений в траекториях на морских участках;
- ROSE (Rough Outlier Set Extraction) - метод грубого подсчета числа аномальных точек на интервалах траектории;
- TN-Outlier (Trajectory-Neighbor based trajectory Outlier) and PN-Outlier (Point-Neighbor-based trajectory Outlier) - ранжирование участков траекторий по количеству ближайших аномальных участков или аномальных точек;
- LDTRAOD (Local Density TRAjectory Outlier Detection) - алгоритм, использующий подсчет метрики схожести интервалов;
- TF-Outlier (Trajectory Fragment Outlier) - алгоритм, использующий подсчет метрики различия интервалов;
- DB-TOD (Driving Behavior-based Trajectory Outlier Detection) - модель обучения с подкреплением на основе принципа максимальной энтропии;
- Hierarchical Clustering - алгоритм, использующий иерархическую кластеризацию по расстоянию между последовательностями точек траектории (edit distance);
- kAA (k-ahead Artificial Arcs) - построение графа точек траекторий и поиск кратчайшего пути;
- LoTAD (Long-term Traffic Anomaly Detection) - алгоритм на основе подсчета Manhattan distance;
- F-DBSCAN - Fusing DBSCAN по историческим данным для построения фичей и последующей классификации траекторий;
- STN-Outlier (Sub-trajectory and Trajectory Neighbor-based Outlier) - построение intra- и inter- фичей по историческим данным, характеризующих отличия в пространственных точках и в направлении траекторий;
- TODCSS (Trajectory Outlier Detection based on Common Slices Sub-sequences) - построение срезов путем соединения последовательных однонаправленных сегментов и и подсчет числа сегментов между соседними срезами;
- ANPR (Automatic Number-Plate Recognition) - классификация траекторий на основе условной системы разбиения на участки (без привязки к данным GPS);
- DS-Traj (Dempster-Shafer for Trajectory outliers) - алгоритм оценки правдоподобия траекторий, основанный на теории Демпстера-Шафера;
- CaD (Context-aware Distance-based algorithm) - алгоритм, использующий TSA (Trend Segmentation Algorithm) сегментацию и PAM (Partition Around Medoids) кластеризацию;
- kNN-LOF - гибридный алгоритм, использующий метрику локальной доступности (local reachability);
- LoOP-AF (Local Outlier Probability for Airline Flights) - алгоритм, использующий вероятностные оценки для LOF (local outlier factor);
- SCG-SFL (Sparse Coding Guided Spatiotemporal Feature Learning) - алгоритм поиска аномалий с использованием сверточной сети на данных в трех измерениях.


Основные используемые подходы:
- Distance-based алгоритмы используют вычисление расстояний между соседними точками, такие алгоритмы эффективны для определения похожести траекторий. Точность таких алгоритмов меньше, потому что они более ориентированы на выявление локальных аномалий, отклонений между близкими траекториями.
- Density-based алгоритмы используют методы кластеризации на основе пространственной плотности (DBSCAN). Эти алгоритмы эффективны для анализа траекторий с небольшим количеством точек поскольку на больших объемах данных работают медленно.
- Pattern mining-based алгоритмы преобразуют массив траекторий в массив транзакций для адаптации к различным алгоритмам поиска шаблонов (pattern mining). Они могут использовать различные метрики, в том числе нехарактерные для геоданных, за счет чего могут давать хорошую точность, однако они не обладают высокой производительностью.
- Алгоритмы, использующие методы машинного обучения, такие как SVM, PCA, уменьшение размерности, а также различные их ансамбли. Недостатком для таких методов отмечают необходимость наличия (построения) разметки по точкам траектории.
- Алгоритмы скоринга, которые выделяют выбросы сравнивая для траекторий различные метрики. Результат сравнения, как правило, существенно зависит от масштаба. Выбор метрик не всегда поддается интерпретации. Стандартными считаются такие метрики как Евклидово расстояние, мера Хаусдорфа, длина наибольшей общей последовательности, динамика по временной шкале (DTW), и другие метрики, основанные на сравнении направлений, углов, скоростей и плотности.


В [7] представлено сравнение 10 моделей:
- Distance-based: TRAOD;
- Density-based: LoTAD, PN-Outlier, TN-Outlier, TF-Outlier, STN-Outlier;
- Pattern mining-based: iBAT, iBOAT, iBDD, MANTRA.


По производительности:

Pattern mining-based - требуют больших вычислительных ресурсов и памяти для хранения паттернов, они эффективны для анализа длинных траекторий.

Density- and distance-based - более быстрые, но зачастую требуют большое количество вычислений попарного расстояния между участками траекторий, они эффективны для анализа большого количества коротких сегментов.

По точности (F-measure, ROC AUC):

Pattern mining-based показали лучшие метрики, потому что эти алгоритмы исследуют различные корреляции между траекториями для поиска аномалий, в то время как  density- and distance-based только вычисляют сходство траекторий.

Выводы по сравнению алгоритмов:

iBOAT - лучший по точности, этот алгоритм выявляет не только аномалии в траекториях, но и участки, которые эти аномалии вызывают.
PN-Outlier - один из лучших по точности среди density- and distance-based алгоритмов, потому что в основе его вычисление поточечного попарного расстояния между участками траекторий.

Для realtime задач, не требующих высокой точности, лучше подходят density- and distance-based алгоритмы. Если требуется высокое качество и точность предсказания, следует использовать pattern mining–based алгоритмы.

Источники:
1. Никита Серенко. Как мы фрод из избы выносили, https://habr.com/ru/company/citymobil/blog/486870/, 2020 год.
2. Вакансия. Разработчик бэкенда в отдел антифрода Такси, https://yandex.ru/jobs/vacancies/разработчик-бэкенда-в-отдел-
антифрода-такси-4722, 2021.
3. Никита Башун. Создание системы антифрода в такси с нуля, https://habr.com/ru/post/512752, 2020.
4. Блог Jump Taxi. Водители-мошенники: как таксопарку обезопасить себя от фрода, https://blog.jump.taxi/fraudtaxidrivers.
5. Блог Lyft. Lyft’s Commitment to Safety, https://www.lyft.com/blog/posts/lyfts-commitment-to-safety, 2019.
6. Блог Uber. Fraud Detection: Using Relational Graph Learning to Detect Collusion, https://eng.uber.com/fraud-detection, 2021.
7. Asma Belhadi, Youcef Djenouri, Jerry Chun-Wei Lin, and Alberto Cano. 2020. Trajectory Outlier Detection: Algorithms, Taxonomies, Evaluation, and Open Challenges. ACM Trans. Manage. Inf. Syst. 11, 3, Article 16 (August 2020), 29 pages. DOI:https://doi.org/10.1145/3399631
8. Manish Gupta, Jing Gao, Charu C. Aggarwal, and Jiawei Han. 2014. Outlier detection for temporal data: A survey. IEEE Transactions on Knowledge and Data Engineering 26, 9 (2014), 2250–2267.
9. Erich Schubert, Arthur Zimek, and Hans-Peter Kriegel. 2014. Local outlier detection reconsidered: A generalized view on locality with applications to spatial, video, and network outlier detection. Data Mining and Knowledge Discovery 28, 1 (2014), 190–237.
10. Yu Zheng, Licia Capra, Ouri Wolfson, and Hai Yang. 2014. Urban computing: Concepts, methodologies, and applications. ACM Transactions on Intelligent Systems and Technology 5, 3 (2014), 38.
11. Yu Zheng. 2015. Trajectory data mining: An overview. ACM Transactions on Intelligent Systems and Technology 6, 3 (2015), 29.
12. Zhenni Feng and Yanmin Zhu. 2016. A survey on trajectory data mining: Techniques and applications. IEEE Access 4 (2016), 2056–2067.
13. Asma Belhadi, Youcef Djenouri, and Jerry Chun-Wei Lin. 2019. Comparative study on trajectory outlier detection algorithms. In Proceedings of the International Conference on Data Mining Workshops. 415–423.
14. Youcef Djenouri, Asma Belhadi, Jerry Chun-Wei Lin, Djamel Djenouri, and Alberto Cano. 2019. A survey on urban traffic anomalies detection algorithms. IEEE Access 7 (2019), 12192–12205.
15. Nawaf O. Alsrehin, Ahmad F. Klaib, and Aws Magableh. 2019. Intelligent transportation and control systems using data mining and machine learning techniques: A comprehensive study. IEEE Access 7 (2019), 49830–49857.
16. Raghavendra Chalapathy and Sanjay Chawla. 2019. Deep learning for anomaly detection: A survey. arXiv:1901.03407.
