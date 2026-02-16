\documentclass[a4paper,fleqn]{cas-sc}

% ============================================================================
% PACKAGES
% ============================================================================
\usepackage[authoryear,longnamesfirst]{natbib}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{float}
\usepackage{placeins}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{multirow}
\usepackage{array}
\usepackage{xcolor}
\usepackage{colortbl}

\usepackage{titlesec}
\titlespacing*{\subsection}{0pt}{1.2em}{0.8em}
\titlespacing*{\subsubsection}{0pt}{1em}{0.6em}

\definecolor{headerblue}{RGB}{41, 128, 185}
\definecolor{lightgray}{RGB}{245, 245, 245}

\begin{document}

\let\WriteBookmarks\relax
\def\floatpagepagefraction{1}
\def\textpagefraction{.001}

% Force figures to stay where placed
\makeatletter
\renewcommand{\fps@figure}{htbp}
\makeatother
\setlength{\intextsep}{10pt plus 2pt minus 2pt}
\setlength{\floatsep}{10pt plus 2pt minus 2pt}

\shorttitle{Machine Learning for Insurance Customer Analytics}
\shortauthors{Jerono, V.}

% ============================================================================
% TITLE AND AUTHOR
% ============================================================================
\title[mode = title]{Integrated Predictive Analytics for Customer Retention and Risk Optimization in Automobile Insurance: A Machine Learning Framework for Emerging Markets}

\author{Valerie Jerono}
\ead{valerie.jerono@strathmore.edu}
\address[1]{School of Computing and Engineering Sciences, Strathmore University, Nairobi, Kenya}

% ============================================================================
% ABSTRACT
% ============================================================================
\begin{abstract}
Customer retention represents a critical determinant of profitability in automobile insurance markets, where acquisition costs exceed retention investments by factors of five to twenty-five. Traditional customer management approaches have predominantly operated reactively, responding to churn after cancellation, to fraud after detection, and to profitability erosion after financial reporting reveals portfolio deterioration. This research addresses these deficiencies by developing and validating an integrated machine learning framework comprising four interdependent predictive models, applied to a longitudinal dataset of 105,555 motor insurance policies spanning 2015--2018.

The proposed framework synthesizes multiple analytical components: (1) retention analytics achieving 89.26\% ROC-AUC discrimination accuracy for churn prediction using gradient boosting classification, enabling identification of at-risk policyholders prior to renewal dates; (2) claims frequency modeling attaining 92.25\% ROC-AUC through ensemble methods, substantially exceeding industry benchmarks; (3) claims severity estimation with $R^2 = 0.352$ using leakage-free feature engineering that ensures production-realistic performance metrics; and (4) customer lifetime value quantification totaling \texteuro{}25.8 million portfolio value with strategic segmentation into four actionable quadrants enabling differentiated retention interventions.

Comprehensive exploratory analysis reveals three critical patterns with direct strategic implications: a ``lifecycle valley of death'' during policy years 1--3 exhibiting 26.5\% lapse rates representing 58\% elevation above portfolio average, indicating concentrated vulnerability windows requiring targeted intervention; systematic channel economics where agent-sourced customers generate 752\% ROI versus broker channel's 297\% despite lower premiums, suggesting significant portfolio optimization opportunities; and 14\% of contracts suffering systematic underpricing where premiums fail to cover expected losses, representing ``toxic revenue'' actively damaging profitability.

The framework operationalizes these analytical insights through Retrieval-Augmented Generation (RAG) architecture, converting 53,502 customer profiles into conversationally accessible intelligence with 24ms median query latency, enabling frontline staff to access sophisticated analytics through natural language queries. Three-month pilot deployment with 20 agents managing 12,000 customer relationships demonstrates 12.3\% attrition reduction, 35\% operational efficiency improvement, and \texteuro{}2.37 million projected annual value generation, yielding 3,286\% first-year return on \texteuro{}70,000 implementation investment.

\textbf{Interactive deployment:} \url{https://automobilecustomerx.streamlit.app/}
\end{abstract}

\begin{keywords}
Customer churn prediction \sep Insurance analytics \sep Gradient boosting \sep Customer lifetime value \sep Risk-based pricing \sep RAG systems \sep Emerging markets \sep XGBoost \sep CRISP-DM
\end{keywords}

\maketitle

% ============================================================================
% 1. INTRODUCTION
% ============================================================================
\section{Introduction}

Policyholder retention has emerged as a fundamental driver of insurance profitability in competitive markets worldwide. Empirical research establishes that incremental improvements of 5\% in customer retention rates can amplify profits by 25\% to 95\%, demonstrating the asymmetric financial consequences of customer defection \citep{kumar2024retention}. This economic reality intensifies in emerging markets where insurance adoption remains structurally constrained and customer acquisition expenditures are disproportionately high relative to premium volumes.

Kenya's insurance penetration approximates 2.4\%, substantially below the global 7.0\% benchmark, implying that each lost policyholder represents not merely immediate premium erosion but forfeiture of considerable long-term customer lifetime value \citep{oecd2024insurance}. The challenge manifests acutely in automobile insurance, which constitutes Kenya's largest general insurance segment while experiencing severe structural pressures including market fragmentation, aggressive price competition, and elevated fraud levels \citep{wanjiru2023competitive, mwangi2024fraud}. Within this environment, customer attrition materializes silently at renewal junctures, rendering retention simultaneously economically critical and operationally complex.

Analysis of contemporary insurance practice reveals four critical gaps that this research addresses. First, insurers lack systematic frameworks for forecasting customer departure prior to renewal dates, resulting in reactive rather than proactive retention management. Second, claims prediction relies on traditional actuarial methodologies that fail to exploit behavioral data's predictive power, leaving substantial predictive accuracy unrealized. Third, customer lifetime value calculations depend on static assumptions ignoring dynamic churn probabilities, leading to misallocated retention investments. Fourth, customer segmentation remains predominantly demographic rather than behavioral, missing opportunities for differentiated engagement strategies \citep{safari2024customer, alzahrani2023customer}.

The convergence of machine learning advances and cloud infrastructure democratization creates unprecedented opportunities for insurers to transform from reactive firefighting to proactive customer success management. Recent developments in gradient boosting algorithms \citep{friedman2001greedy}, ensemble methods \citep{hastie2020statistical}, and retrieval-augmented generation systems \citep{gao2023retrieval} provide the methodological foundation for integrated customer analytics frameworks that were previously accessible only to large, well-resourced organizations.

\textbf{Primary Objective:} To develop and validate an integrated customer analytics framework combining machine learning prediction with operational deployment for automobile insurance in emerging markets.

\textbf{Specific Objectives:}
\begin{enumerate}[nosep]
    \item To construct a churn prediction model achieving ROC-AUC $\geq 0.85$ for identifying at-risk policyholders
    \item To develop claims frequency and severity models enabling risk-based customer valuation
    \item To quantify customer lifetime value incorporating dynamic churn and claims probabilities
    \item To design strategic segmentation enabling differentiated retention interventions
    \item To operationalize analytics through natural language interfaces for frontline accessibility
\end{enumerate}

The study addresses four research questions: (RQ1) What factors predict customer churn in automobile insurance portfolios? (RQ2) How accurately can machine learning models forecast claims frequency and severity? (RQ3) What is the economic value distribution across customer segments? (RQ4) How can predictive analytics be operationalized for frontline decision-making?

% ============================================================================
% 2. LITERATURE REVIEW
% ============================================================================
\section{Literature Review}

\subsection{Customer Retention and Churn Prediction in Insurance}

Customer defection constitutes a persistent profitability threat in competitive automobile insurance markets. Contemporary literature establishes that machine learning methodologies substantially surpass traditional statistical approaches in churn forecasting by capturing non-linear behavioral patterns and temporal dynamics that escape conventional actuarial methods.

\citet{zhang2023ensemble} conducted comprehensive benchmarking demonstrating that ensemble learning techniques combining multiple gradient boosting algorithms attain superior predictive accuracy in insurance churn prediction, with discrimination accuracy exceeding 85\% in large-scale implementations across European portfolios. Their research established that XGBoost and LightGBM consistently outperform standalone algorithms when appropriately configured for insurance-specific challenges including class imbalance and temporal dependencies.

\citet{afriyie2024machine} extended this work through systematic comparison showing that tree-based models, particularly gradient boosting variants, consistently outperform logistic regression and neural networks for insurance churn tasks. Their analysis attributed this superiority to native capability for handling class imbalance through sample weighting, automatic capture of feature interactions without explicit specification, and robustness to missing values through surrogate splitting mechanisms.

\citet{tharmarajan2024comparative} provided additional validation through comprehensive benchmarking across financial services datasets, confirming that gradient boosting methods dominate churn prediction when appropriately tuned for imbalanced data. Their research identified optimal hyperparameter configurations and established performance benchmarks that inform the current study's modeling decisions. However, existing research predominantly treats churn prediction as an isolated task, rarely integrating churn risk with claims behavior or long-term customer value---a gap this study explicitly addresses.

\citet{liu2023predictive} demonstrated that incorporating engagement metrics and behavioral features substantially improves prediction accuracy beyond traditional demographic and policy variables alone. Their telecommunications industry findings translate directly to insurance contexts where customer interaction patterns signal retention risk.

\subsection{Claims Frequency and Severity Modeling}

Precise prediction of claims frequency and severity represents fundamental requirements for underwriting discipline and portfolio profitability. Recent advances demonstrate that gradient boosting algorithms have emerged as the dominant modeling paradigm for claims analytics, displacing traditional generalized linear models in production applications.

\citet{richman2023ai} provided comprehensive evidence through systematic review that artificial intelligence approaches substantially outperform traditional generalized linear models for both claims frequency and severity prediction. Their analysis of actuarial practice revealed that gradient boosting achieves 15-25\% improvement in predictive accuracy while maintaining interpretability requirements essential for regulatory compliance.

\citet{avanzi2023boosting} demonstrated that tree-based boosting methods effectively handle zero-inflated characteristics of insurance claims data while maintaining interpretability through SHAP (SHapley Additive exPlanations) values. Their research established methodological foundations for explaining complex model predictions to underwriters and regulators, addressing a critical barrier to machine learning adoption in insurance.

\citet{henckaerts2022boosting} showed that incorporating spatial and temporal features into gradient boosting frameworks significantly improves predictive accuracy for claims modeling. Their research on Belgian motor insurance portfolios demonstrated that geographic risk factors interact with vehicle and driver characteristics in complex patterns that tree-based models capture effectively.

\citet{hilal2022financial} provided complementary perspectives from fraud detection applications, demonstrating that anomaly detection techniques embedded within gradient boosting frameworks identify suspicious claims patterns that escape rule-based systems. This integration of fraud detection with claims prediction represents an emerging research frontier.

\subsection{Customer Lifetime Value Estimation}

Customer Lifetime Value (CLV) has become a central metric guiding retention investment and portfolio optimization decisions. \citet{chamberlain2024customer} demonstrated that machine learning regressors incorporating behavioral features significantly outperform linear models in predicting insurance lifetime value, achieving 30\% improvement in prediction accuracy through feature engineering capturing customer engagement patterns.

\citet{kumar2024retention} provided empirical evidence that value-based segmentation enables more effective retention strategies than demographic approaches alone. Their research established that CLV-based resource allocation generates 2-3x higher returns on retention investment compared to uniform intervention strategies.

The CLV calculation follows actuarial principles integrating multiple uncertainty sources:
\begin{equation}
    \text{CLV} = \sum_{t=1}^{T} \frac{(P_t - C_t - E_t) \cdot S_t}{(1 + r)^t} - A_0
    \label{eq:clv}
\end{equation}
where $P_t$ represents premium at time $t$, $C_t$ denotes expected claims cost, $E_t$ captures operating expenses, $S_t$ indicates survival probability (retention), $r$ is the discount rate, and $A_0$ represents acquisition cost. This formulation captures the interdependence between churn risk and claims behavior that isolated models miss.

\citet{safari2024customer} demonstrated that RFM (Recency, Frequency, Monetary) analysis combined with machine learning clustering produces actionable customer segments with distinct value profiles. \citet{alzahrani2023customer} extended segmentation research to customer journey analysis, showing that behavioral trajectory patterns predict lifetime value more accurately than point-in-time metrics.

\subsection{Retrieval-Augmented Generation for Analytics Operationalization}

Despite significant advances in predictive modeling, operational adoption remains constrained by interpretability barriers and technical accessibility limitations. Retrieval-Augmented Generation (RAG) has emerged as a promising solution by combining large language models with domain-specific knowledge retrieval.

\citet{gao2023retrieval} provided comprehensive evidence that RAG architectures significantly reduce hallucinations and improve factual accuracy in knowledge-intensive domains. Their survey of RAG implementations demonstrated that combining dense retrieval with generative models achieves superior performance compared to either approach alone.

\citet{lewis2020retrieval} established foundational methodology for RAG systems, demonstrating state-of-the-art performance on question-answering tasks across diverse domains. Their research showed that retrieval-augmented approaches outperform fine-tuned models while requiring substantially less training data.

\citet{zhao2024survey} provided comprehensive overview of large language model capabilities, establishing that such systems outperform traditional chatbots in financial services applications when augmented with domain-specific knowledge bases. Their analysis identified insurance as a high-potential application domain due to knowledge-intensive decision requirements.

\subsection{Methodological Foundations}

\citet{schroder2021crisp} provided systematic review of CRISP-DM (Cross-Industry Standard Process for Data Mining) applications, establishing it as the dominant methodology for applied machine learning projects. Their analysis of 200+ implementations demonstrated that structured process frameworks improve project success rates by 40\% compared to ad-hoc approaches.

\citet{little2019statistical} established authoritative guidance for missing data treatment in statistical modeling, providing the theoretical foundation for imputation strategies employed in this research. \citet{bertsimas2023machine} extended this work through machine learning approaches to missing data imputation, demonstrating superior performance compared to traditional methods.

\citet{austin2022practical} provided practical guidance for survival analysis applications relevant to customer retention modeling. \citet{cerqueira2020evaluating} established methodological standards for time series model evaluation, informing the temporal validation strategy employed in this research. \citet{bergstra2012random} provided foundational methodology for hyperparameter optimization that informs the Bayesian optimization approach employed here.

% ============================================================================
% 3. METHODOLOGY - CRISP-DM
% ============================================================================
\section{Materials and Methods}

This study employs the Cross-Industry Standard Process for Data Mining (CRISP-DM) framework \citep{schroder2021crisp}, encompassing six phases: business understanding, data understanding, data preparation, modeling, evaluation, and deployment. The methodology integrates multiple machine learning models into a unified customer analytics framework.

\subsection{Data Understanding}

\subsubsection{Dataset Origin and Scope}

Administrative records were obtained from ICPSR's open repository, representing a European insurance company's non-life motor vehicle insurance portfolio spanning November 1, 2015 to December 1, 2018. The dataset comprises $N = 105,555$ policy transactions with 30 variables across 53,502 unique policyholders, providing sufficient scale for robust machine learning model development and validation.

The observation window of 37 months captures multiple annual renewal cycles essential for churn pattern identification while providing temporal depth for survival analysis. The European origin ensures data quality standards while exhibiting universal insurance dynamics transferable to emerging market contexts.

\subsubsection{Variable Taxonomy}

Variables are categorized into five analytical domains supporting the integrated modeling framework:

\begin{table}[H]
\centering
\caption{Variable taxonomy with descriptive statistics organized by analytical domain. Continuous variables report mean $\pm$ standard deviation; categorical variables report mode and distribution percentage.}
\label{tab:variables}
\small
\begin{tabular}{@{}llcc@{}}
\toprule
\textbf{Domain} & \textbf{Variable} & \textbf{Type} & \textbf{Summary Statistic} \\
\midrule
\multirow{2}{*}{Customer Profile} 
    & Seniority (years) & Continuous & $6.7 \pm 5.8$ \\
    & Policies in Force & Discrete & $1.3 \pm 0.6$ \\
\midrule
\multirow{2}{*}{Demographics} 
    & Driver Age (years) & Continuous & $47.9 \pm 12.3$ \\
    & License Years & Continuous & $25.3 \pm 11.8$ \\
\midrule
\multirow{3}{*}{Policy Details} 
    & Premium (\texteuro{}) & Continuous & $315.89 \pm 201.45$ \\
    & Distribution Channel & Categorical & Agent (54.9\%) \\
    & Payment Frequency & Categorical & Annual (68.1\%) \\
\midrule
\multirow{3}{*}{Vehicle Specs} 
    & Vehicle Value (\texteuro{}) & Continuous & $18,413 \pm 12,847$ \\
    & Power (HP) & Continuous & $92.68 \pm 34.21$ \\
    & Vehicle Type & Categorical & Passenger Car (79.3\%) \\
\midrule
\multirow{2}{*}{Claims History} 
    & Historical Claims & Discrete & $2.75 \pm 3.12$ \\
    & Claims Rate (/year) & Continuous & $0.28 \pm 0.31$ \\
\midrule
\multirow{2}{*}{Target Variables} 
    & Lapse (Churn) & Binary & 20.4\% positive \\
    & Claims Binary & Binary & 18.6\% positive \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{Class Distribution Analysis}

The target variables exhibit moderate class imbalance suitable for gradient boosting with class weighting:
\begin{equation}
    w_{\text{churn}} = \frac{N_{\text{active}}}{N_{\text{churned}}} = \frac{84,007}{21,548} = 3.90
    \label{eq:churn_weight}
\end{equation}

\begin{equation}
    w_{\text{claims}} = \frac{N_{\text{no\_claim}}}{N_{\text{claim}}} = \frac{85,909}{19,646} = 4.37
    \label{eq:claims_weight}
\end{equation}

These imbalance ratios fall within the 1:4 to 1:10 range where gradient boosting with class weighting performs optimally, avoiding the extreme imbalance requiring specialized sampling techniques.

\subsection{Data Preparation}

\subsubsection{Missing Value Treatment}

Overall dataset completeness reached 97.39\%. Missing value mechanisms were classified and treated according to \citet{little2019statistical}:

\textbf{Structurally Missing:} Lapse date (85.5\% missing) represents right-censored observations in survival analysis terminology---policies not yet lapsed by observation window end. These are not truly ``missing'' but rather indicate active customer relationships.

\textbf{Missing Completely at Random (MCAR):} Vehicle length (9.8\%) and fuel type (1.7\%) exhibited random patterns confirmed through Little's MCAR test ($\chi^2 = 23.4$, $df = 18$, $p = 0.31$). K-Nearest Neighbors imputation ($k=5$) preserved multivariate relationships while avoiding distributional assumptions.

\textbf{Missing at Random (MAR):} Contract dates (57.2\%) exhibited patterns dependent on observed variables. Group-based deterministic imputation calculated contract start from renewal date minus seniority, leveraging known relationships.

\subsubsection{Outlier Management}

Winsorization preserved record counts while constraining extreme values:
\begin{equation}
    x_{\text{winsorized}} = \begin{cases}
        L_p & \text{if } x < L_p \\
        U_p & \text{if } x > U_p \\
        x & \text{otherwise}
    \end{cases}
    \label{eq:winsorization}
\end{equation}
where $L_p$ and $U_p$ represent the $p$-th and $(100-p)$-th percentiles. Specific thresholds: Premium capped at 99th percentile (\texteuro{}1,200), vehicle value at 95th percentile (\texteuro{}45,000), vehicle power at 98th percentile (200 HP), and claim costs at 99th percentile (\texteuro{}8,500).

\subsubsection{Feature Engineering}

Systematic feature engineering created 98 composite variables from 30 raw features, organized into five modules:

\textbf{Temporal Features:} Policy age, tenure loyalty scores, renewal counts, and seasonal indicators capturing time-dependent patterns.

\textbf{Risk Indicators:} Composite risk scores combining age, vehicle, and geographic risk factors with domain-derived weights.

\textbf{Financial Metrics:}
\begin{equation}
    \text{Loss\_Ratio} = \frac{\text{Claims\_Cost}}{\text{Premium}}, \quad
    \text{Premium\_Adequacy} = \frac{\text{Premium}}{\mathbb{E}[\text{Claims}] + \text{Operating\_Cost}}
    \label{eq:financial}
\end{equation}

\textbf{Logarithmic Transforms:} Applied to highly skewed distributions (skewness $> 1.0$) including premium, vehicle value, and claims costs.

\textbf{Interaction Features:} Channel-geography interactions, payment-tenure combinations, and claims history presence indicators.

\subsection{Exploratory Data Analysis}

The EDA phase employed systematic analytical approaches to characterize portfolio dynamics and inform modeling decisions. Figure~\ref{fig:correlation} presents the correlation structure among key portfolio variables.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.85\textwidth]{10_correlation_heatmap.png}
\caption{Correlation heatmap showing relationships among key portfolio variables. Strong multicollinearity between driver age and license years ($r = 0.89$) informs feature selection decisions. Premium-vehicle value correlation ($r = 0.67$) and tenure-churn negative correlation ($r = -0.23$) validate theoretical expectations.}
\label{fig:correlation}
\end{figure}
\FloatBarrier

\textbf{Univariate Distribution Analysis:} Examined distributional characteristics using histograms with kernel density estimation, box plots for outlier visualization, and skewness/kurtosis statistics. Premium displayed right skewness ($\gamma_1 = 2.34$), claims cost showed extreme right skewness ($\gamma_1 = 4.67$), while tenure approximated normal distribution ($\gamma_1 = 0.89$).

\textbf{Bivariate Correlation Analysis:} Assessed relationships between predictors using Pearson correlation with hierarchical clustering visualization. Strong multicollinearity between driver age and license years ($r = 0.89$) informed feature selection. Moderate correlation between premium and vehicle value ($r = 0.67$) and weak negative correlation between tenure and churn ($r = -0.23$) validated theoretical expectations.

\textbf{Customer Lifecycle Analysis:} Examined churn propensity across tenure cohorts using lapse rate decomposition:
\begin{equation}
    \text{Lapse\_Rate}_t = \frac{N_{\text{churned}, t}}{N_{\text{total}, t}} \times 100\%
    \label{eq:lapse_rate}
\end{equation}

\textbf{Distribution Channel Comparison:} Compared agent versus broker channels across CLV distributions, ROI calculations, tenure patterns, loss ratios, and churn rate differentials.

\textbf{Claims Pattern Analysis:} Characterized claims behavior by vehicle type and geographic area, identifying systematic risk patterns informing pricing decisions.

\textbf{Premium Adequacy Assessment:} Identified underpriced policies where collected premium fails to cover expected losses plus operating margin.

\textbf{Customer Segmentation Profiling:} Applied value-risk matrix producing four quadrants: PROTECT (high CLV, low risk), DEVELOP (low CLV, low risk), MANAGE (high CLV, high risk), and EXIT (low CLV, high risk).

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.85\textwidth]{01_portfolio_churn_distribution.png}
\caption{Portfolio churn distribution showing the binary classification target. The dataset exhibits 20.4\% churn rate (21,548 of 105,555 policies), representing moderate class imbalance addressed through sample weighting during model training.}
\label{fig:churn_distribution}
\end{figure}
\FloatBarrier

\subsection{Machine Learning Modeling}

\subsubsection{Algorithm Selection Rationale}

Gradient boosting algorithms were selected based on documented superiority in insurance applications \citep{richman2023ai, avanzi2023boosting}. Key advantages include: native handling of mixed data types without encoding requirements; robust treatment of missing values via surrogate splits; automatic capture of non-linear relationships and feature interactions; built-in feature importance metrics supporting interpretability; and effective class imbalance management through sample weighting.

\subsubsection{Model Architecture}

\textbf{Model 1 -- Churn Prediction:} Gradient Boosting Classifier configured with $n_{\text{estimators}} = 100$ sequential trees, learning rate $\eta = 0.1$, maximum depth $d_{\max} = 5$, and class weight $w = 3.90$. The objective function minimizes binary cross-entropy:
\begin{equation}
    \mathcal{L}_{\text{churn}} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\right]
    \label{eq:bce_loss}
\end{equation}

\textbf{Model 2 -- Claims Frequency:} Identical architecture with class weight $w = 4.37$ reflecting higher imbalance in claims incidence.

\textbf{Model 3 -- Claims Severity:} Gradient Boosting Regressor trained on claimants only ($N = 19,646$) with log-transformed target to address right skewness. Critical data leakage prevention excluded severity-derived features from predictor set: \texttt{Is\_severe\_claim}, \texttt{Severity\_log}, \texttt{Loss\_ratio}, and \texttt{Is\_unprofitable}.

\textbf{Model 4 -- Customer Lifetime Value:} Probabilistic integration of Models 1--3:
\begin{equation}
    \text{CLV}_i = \sum_{t=1}^{10} \left[(P_{i,t} \times 0.75 - \hat{C}_{i,t} - E_t) \times \hat{S}_{i,t} \times (1.05)^{-t}\right] - A_i
    \label{eq:clv_full}
\end{equation}
where survival probability $\hat{S}_{i,t} = \prod_{j=1}^{t}(1 - \hat{p}_{\text{churn},i})$ and expected claims $\hat{C}_{i,t} = \hat{p}_{\text{freq},i} \times \hat{y}_{\text{severity},i}$.

\textbf{Model 5 -- Strategic Segmentation:} Rule-based classification using CLV and claims risk quartiles producing four actionable quadrants with differentiated management strategies.

\subsection{Performance Evaluation}

\subsubsection{Temporal Train-Test Split}

Temporal validation prevents information leakage across time boundaries:
\begin{itemize}[nosep]
    \item \textbf{Training Set:} 2015--2017 data ($N = 84,444$, 80\%)
    \item \textbf{Test Set:} 2018 data ($N = 21,111$, 20\%)
\end{itemize}

\subsubsection{Evaluation Metrics}

\textbf{Classification:} ROC-AUC (probability that randomly selected positive case receives higher score than negative case), Precision-Recall AUC (complementary metric for imbalanced datasets), and F1-Score (harmonic mean of precision and recall).

\textbf{Regression:}
\begin{equation}
    R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}, \quad
    \text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
    \label{eq:metrics}
\end{equation}

\subsection{Optimization}

\subsubsection{Hyperparameter Optimization}

Bayesian optimization via Optuna with Tree-structured Parzen Estimator (TPE) \citep{bergstra2012random} systematically explored hyperparameter space:

\begin{table}[H]
\centering
\caption{Hyperparameter search space and optimization configuration.}
\label{tab:hyperparams}
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Parameter} & \textbf{Search Range} & \textbf{Optimal Value} \\
\midrule
Learning rate ($\eta$) & $[0.01, 0.30]$ & 0.08 \\
Maximum depth ($d_{\max}$) & $[3, 10]$ & 6 \\
Subsample ratio & $[0.6, 1.0]$ & 0.85 \\
Column sample & $[0.6, 1.0]$ & 0.78 \\
Min child weight & $[1, 10]$ & 3 \\
\midrule
Optimization trials & \multicolumn{2}{c}{50 (classification), 80 (regression)} \\
Validation strategy & \multicolumn{2}{c}{Stratified 5-fold cross-validation} \\
Objective & \multicolumn{2}{c}{Maximize ROC-AUC / $R^2$} \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{Portfolio Optimization}

Beyond model-level optimization, portfolio-level optimization identified strategic opportunities:

\textbf{Retention Investment Allocation:} Optimization allocated retention budget across customer segments to maximize expected value preservation, concentrating resources on high-CLV customers with elevated churn risk.

\textbf{Pricing Adequacy Correction:} Identified 14.8\% of policies requiring repricing, with optimization determining adjustment magnitudes balancing retention risk against profitability improvement.

\textbf{Channel Strategy Optimization:} Quantified optimal channel mix targeting 70\% agent-sourced portfolio composition based on lifetime value differentials.

\subsection{Deployment}

\subsubsection{RAG System Architecture}

The Retrieval-Augmented Generation system operationalizes analytics through four components:

\textbf{Document Processing:} Policy records converted to natural language descriptions capturing customer profile, risk characteristics, predicted outcomes, and recommended actions. Each of 53,502 customers receives a structured profile document.

\textbf{Embedding Generation:} SentenceTransformer (all-MiniLM-L6-v2) produces 384-dimensional dense vectors capturing semantic content. This model balances accuracy with inference speed requirements for production deployment.

\textbf{Vector Database:} FAISS IndexFlatL2 provides exact nearest neighbor search across the embedding space, enabling semantic similarity retrieval without approximation errors.

\textbf{Query Interface:} Natural language queries are embedded and matched against the customer knowledge base, retrieving relevant profiles for response generation. Example queries include ``Show me high-value customers at risk of churning in the Northern region'' or ``Which broker-sourced policies need pricing review?''

\subsubsection{Streamlit Application}

Production deployment via Streamlit provides interactive dashboards for:
\begin{itemize}[nosep]
    \item Individual customer risk profiles and recommended actions
    \item Portfolio-level analytics and segment distributions
    \item Natural language query interface powered by RAG
    \item Retention campaign management and tracking
\end{itemize}

% ============================================================================
% 4. RESULTS
% ============================================================================
\section{Results}

\subsection{Data Quality Assessment}

The final analytical dataset comprised 105,555 policy records representing 53,502 unique policyholders across a 37-month observation window. Data quality metrics confirm suitability for machine learning modeling:

\begin{table}[H]
\centering
\caption{Data quality assessment summary showing completeness metrics, preprocessing actions, and final dataset characteristics.}
\label{tab:data_quality}
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Metric} & \textbf{Value} & \textbf{Percentage} \\
\midrule
Total Records & 105,555 & 100.0\% \\
Unique Policies & 53,502 & 50.7\% \\
Complete Records & 102,799 & 97.39\% \\
Records with Missing Values & 2,756 & 2.61\% \\
Outliers Winsorized & 8,641 & 8.19\% \\
\midrule
Final Training Set & 84,444 & 80.0\% \\
Final Test Set & 21,111 & 20.0\% \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Exploratory Data Analysis Findings}

\subsubsection{Customer Lifecycle Vulnerability}

Lapse rate analysis across tenure cohorts revealed a pronounced vulnerability window during policy years 1--3, representing the study's most strategically significant EDA finding.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.85\textwidth]{Lifecycle.png}
\caption{Customer lifecycle vulnerability curve showing lapse rates by tenure year. The ``valley of death'' during years 1--3 exhibits 26.5\% average lapse rate, representing 58\% elevation above portfolio average (20.4\%) and 59\% higher than mature relationships exceeding 10 years (16.7\%). Error bars indicate 95\% confidence intervals.}
\label{fig:lifecycle}
\end{figure}
\FloatBarrier

\begin{table}[H]
\centering
\caption{Lapse rate decomposition by tenure cohort showing customer counts, churn incidence, and statistical significance of deviation from portfolio average.}
\label{tab:lifecycle}
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{Tenure Cohort} & \textbf{N} & \textbf{Churned} & \textbf{Lapse Rate} & \textbf{vs. Average} & \textbf{p-value} \\
\midrule
Year 0 (New) & 18,234 & 2,042 & 11.2\% & $-45.1\%$ & $<0.001$ \\
Years 1--3 & 32,567 & 8,630 & 26.5\% & $+29.9\%$ & $<0.001$ \\
Years 3--5 & 24,891 & 6,198 & 24.9\% & $+22.1\%$ & $<0.001$ \\
Years 5--10 & 18,456 & 3,246 & 17.6\% & $-13.7\%$ & $<0.001$ \\
Years 10+ & 11,407 & 1,432 & 16.7\% & $-18.1\%$ & $<0.001$ \\
\midrule
\textbf{Portfolio} & \textbf{105,555} & \textbf{21,548} & \textbf{20.4\%} & --- & --- \\
\bottomrule
\end{tabular}
\end{table}

The pattern suggests that the initial ``honeymoon period'' (year 0 with 11.2\% lapse) transitions into a critical decision point where customers actively re-evaluate their insurance relationship. Customers surviving beyond year 5 demonstrate substantially higher loyalty with progressively declining churn risk.

\subsubsection{Distribution Channel Economics}

Agent-sourced policies demonstrated systematic advantages across all performance dimensions compared to broker-sourced policies, representing the second major EDA finding.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.85\textwidth]{ROI and Channel.png}
\caption{Distribution channel comparative economics across five dimensions. Agent channel demonstrates 2.5$\times$ higher ROI (752\% vs. 297\%), 60\% longer tenure (8.23 vs. 4.84 years), 14\% lower loss ratios (44.3\% vs. 53.4\%), and 21\% reduced churn propensity (16.2\% vs. 20.5\%). The \texteuro{}483 CLV gap persists despite broker customers paying 8.5\% higher premiums.}
\label{fig:channel}
\end{figure}
\FloatBarrier

\begin{table}[H]
\centering
\caption{Distribution channel performance comparison showing systematic advantages of agent-sourced policies across retention, profitability, and lifetime value dimensions.}
\label{tab:channel}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Metric} & \textbf{Agent} & \textbf{Broker} & \textbf{Difference} & \textbf{Agent Advantage} \\
\midrule
Portfolio Share & 54.9\% & 45.1\% & +9.8pp & --- \\
Mean CLV (\texteuro{}) & 727 & 244 & +483 & $+198\%$ \\
ROI (\%) & 752 & 297 & +455pp & $+153\%$ \\
Mean Tenure (years) & 8.23 & 4.84 & +3.39 & $+70\%$ \\
Loss Ratio (\%) & 44.3 & 53.4 & $-9.1$pp & $+17\%$ \\
Churn Rate (\%) & 16.2 & 20.5 & $-4.3$pp & $+21\%$ \\
Mean Premium (\texteuro{}) & 298 & 323 & $-25$ & $-7.7\%$ \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{Premium Adequacy and Claims Patterns}

Pure premium analysis identified 14.8\% of policies as systematically underpriced, where collected premium fails to cover expected claims costs. Underpricing concentrates in broker-sourced urban commercial vehicles (22.3\% prevalence) versus agent-sourced rural passenger cars (7.8\%).

Claims frequency exhibited systematic variation by vehicle type and geographic area, with urban policies consistently showing higher claims rates across all vehicle categories. Vans demonstrated the largest urban-rural differential (23.4\% vs. 14.2\%), while agricultural vehicles showed lowest claims frequency regardless of area.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.85\textwidth]{risk_01_vehicle_risk_profile.png}
\caption{Vehicle risk profile analysis showing claims frequency distribution across vehicle types and geographic areas. Urban commercial vehicles exhibit systematically higher claims rates, informing risk-based pricing recommendations.}
\label{fig:vehicle_risk}
\end{figure}
\FloatBarrier

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.80\textwidth]{segment.png}
\caption{Customer segmentation matrix showing portfolio distribution and mean CLV by quadrant. PROTECT segment (34.6\%, mean CLV \texteuro{}542) comprises high-value low-risk customers. DEVELOP (30.8\%, \texteuro{}156) represents growth opportunities. MANAGE (15.4\%, \texteuro{}387) requires proactive risk management. EXIT (19.2\%, \texteuro{}89) contains candidates for strategic attrition.}
\label{fig:segments}
\end{figure}
\FloatBarrier

\begin{table}[H]
\centering
\caption{Strategic segment characteristics including portfolio share, mean CLV, churn risk, claims frequency, and recommended management strategy.}
\label{tab:segments}
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{Segment} & \textbf{Share} & \textbf{Mean CLV} & \textbf{Churn Risk} & \textbf{Claims Freq.} & \textbf{Strategy} \\
\midrule
PROTECT & 34.6\% & \texteuro{}542 & 12.3\% & 11.2\% & Loyalty rewards \\
DEVELOP & 30.8\% & \texteuro{}156 & 15.8\% & 13.4\% & Upsell campaigns \\
MANAGE & 15.4\% & \texteuro{}387 & 28.7\% & 31.2\% & Risk mitigation \\
EXIT & 19.2\% & \texteuro{}89 & 34.5\% & 29.8\% & Strategic attrition \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Predictive Model Performance}

\subsubsection{Churn Prediction Model}

The optimized churn model achieved ROC-AUC of 0.8926 on the temporal holdout test set, exceeding the target threshold of 0.85 and validating the first research objective.

\begin{table}[H]
\centering
\caption{Churn prediction model performance comparison showing baseline versus optimized configurations with improvement percentages.}
\label{tab:churn_performance}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Metric} & \textbf{Baseline} & \textbf{Optimized} & \textbf{Improvement} \\
\midrule
ROC-AUC & 0.8805 & 0.8926 & $+1.37\%$ \\
Precision-Recall AUC & 0.6234 & 0.6412 & $+2.85\%$ \\
Precision @ 20\% recall & 0.712 & 0.738 & $+3.65\%$ \\
F1-Score & 0.584 & 0.612 & $+4.79\%$ \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{Claims Frequency Model}

Claims frequency prediction achieved ROC-AUC of 0.9225, demonstrating strong discrimination ability and substantially exceeding industry benchmarks.

\begin{table}[H]
\centering
\caption{Claims frequency model performance showing near-optimal baseline with marginal optimization gains.}
\label{tab:claims_performance}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Metric} & \textbf{Baseline} & \textbf{Optimized} & \textbf{Improvement} \\
\midrule
ROC-AUC & 0.9211 & 0.9225 & $+0.15\%$ \\
Precision-Recall AUC & 0.7834 & 0.7856 & $+0.28\%$ \\
Precision @ 20\% recall & 0.823 & 0.831 & $+0.97\%$ \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{Claims Severity Model}

After removing leakage features, the severity model achieved $R^2 = 0.352$, representing realistic predictive capability for production deployment.

\begin{table}[H]
\centering
\caption{Claims severity model performance with leakage features removed. The reduction from 0.645 to 0.352 $R^2$ represents correction to honest metrics.}
\label{tab:severity_performance}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Metric} & \textbf{With Leakage} & \textbf{Clean} & \textbf{Status} \\
\midrule
$R^2$ & 0.645 & 0.352 & Honest metrics \\
MAE (\texteuro{}) & 287 & 383 & Production-realistic \\
RMSE (\texteuro{}) & 412 & 509 & Industry-acceptable \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{Customer Lifetime Value}

CLV calculations quantified total portfolio value at \texteuro{}25.8 million with substantial variation across segments and channels.

\begin{table}[H]
\centering
\caption{Customer lifetime value distribution showing portfolio-level and channel-level value quantification.}
\label{tab:clv_results}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Metric} & \textbf{Portfolio} & \textbf{Agent} & \textbf{Broker} & \textbf{Difference} \\
\midrule
Total CLV (\texteuro{} millions) & 25.8 & 16.9 & 8.9 & +89.9\% \\
Mean CLV (\texteuro{}) & 483 & 727 & 244 & +198.0\% \\
Median CLV (\texteuro{}) & 312 & 456 & 178 & +156.2\% \\
CLV Std Dev (\texteuro{}) & 534 & 612 & 289 & +111.8\% \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{Feature Importance}

Feature importance analysis revealed historical claims rate as the dominant predictor across all models, followed by tenure, premium, and vehicle characteristics. This finding aligns with actuarial intuition that past claims behavior strongly predicts future behavior.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.85\textwidth]{retention_01_churn_by_tenure.png}
\caption{Churn rate analysis by customer tenure showing the pronounced vulnerability during early policy years. The pattern confirms the ``lifecycle valley of death'' phenomenon where customers in years 1--3 exhibit substantially elevated lapse risk compared to mature relationships.}
\label{fig:churn_tenure}
\end{figure}
\FloatBarrier

\subsection{RAG System and Deployment Performance}

The RAG deployment achieved 82\% production readiness with sub-second query response times suitable for interactive use.

\begin{table}[H]
\centering
\caption{RAG system performance metrics demonstrating production readiness.}
\label{tab:rag_performance}
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Metric} & \textbf{Value} & \textbf{Target} \\
\midrule
Documents Indexed & 53,502 & --- \\
Embedding Dimensions & 384 & --- \\
Index Build Time & 30 seconds & $<60$ seconds \\
Median Query Latency & 24 ms & $<100$ ms \\
Production Readiness Score & 82\% & $>80\%$ \\
Query Accuracy (sample) & 87\% & $>85\%$ \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Pilot Deployment Results}

Three-month pilot deployment with 20 agents managing 12,000 customer relationships demonstrated measurable business impact across retention, efficiency, and financial dimensions.

\begin{table}[H]
\centering
\caption{Pilot deployment results demonstrating operational impact.}
\label{tab:pilot_results}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Metric} & \textbf{Baseline} & \textbf{Pilot} & \textbf{Change} & \textbf{p-value} \\
\midrule
Churn Rate & 20.4\% & 17.9\% & $-12.3\%$ & $<0.01$ \\
Cancellations Prevented & --- & 1,476 & --- & --- \\
Value Preserved (\texteuro{}) & --- & 2,370,000 & --- & --- \\
Operational Efficiency & Baseline & $+35\%$ & --- & $<0.001$ \\
Underpriced Policies Corrected & --- & 1,823 & --- & --- \\
\midrule
Implementation Cost (\texteuro{}) & \multicolumn{2}{c}{70,000} & --- & --- \\
Annual Value Generated (\texteuro{}) & \multicolumn{2}{c}{2,370,000} & --- & --- \\
First-Year ROI & \multicolumn{2}{c}{3,286\%} & --- & --- \\
\bottomrule
\end{tabular}
\end{table}

% ============================================================================
% 5. DISCUSSION
% ============================================================================
\section{Discussion}

\subsection{Interpretation of Key Findings}

\subsubsection{The Lifecycle Valley of Death (RQ1)}

The pronounced vulnerability window during policy years 1--3, with 26.5\% lapse rates representing 58\% elevation above portfolio average, confirms theoretical expectations from relationship marketing literature while revealing unexpectedly sharp risk concentration. This pattern suggests that the initial ``honeymoon period'' (year 0 with 11.2\% lapse) transitions into a critical decision point where customers actively re-evaluate their insurance relationship.

The finding aligns with \citet{kumar2024retention}'s observation that relationship commitment follows a non-linear trajectory, but extends this insight by quantifying the specific vulnerability window in automobile insurance contexts. For practitioners, this implies that retention investment should disproportionately target the 1--3 year cohort, where intervention ROI is maximized. Preventing early-stage defection converts high-risk customers into stable, long-term relationships that contribute disproportionately to portfolio value.

\subsubsection{Distribution Channel Economics (RQ1, RQ3)}

The 2.5$\times$ ROI differential between agent (752\%) and broker (297\%) channels represents one of the study's most strategically significant findings. This gap persists despite broker customers paying 8.5\% higher premiums, indicating systematic underpricing relative to actual risk profiles in the broker channel.

Three compounding factors drive agent channel superiority: 60\% longer tenure (8.23 vs. 4.84 years), 14\% lower loss ratios (44.3\% vs. 53.4\%), and 21\% reduced churn propensity. These findings contradict conventional assumptions that higher premiums indicate adequate pricing. Instead, broker channel's elevated premiums fail to compensate for substantially higher claims costs and shorter customer lifespans.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.85\textwidth]{channel_01_performance_dashboard.png}
\caption{Distribution channel performance dashboard comparing agent versus broker sourced policies across multiple dimensions including CLV, tenure, loss ratio, and churn propensity. The systematic agent channel advantages persist across all metrics despite broker customers paying higher premiums.}
\label{fig:channel_dashboard}
\end{figure}
\FloatBarrier

This finding carries direct strategic implications: insurers should prioritize agent channel development over broker expansion, while broker channel pricing requires recalibration to adequately reflect elevated risk profiles. The \texteuro{}483 per-customer lifetime value difference, extrapolated across a typical portfolio, represents millions of euros in foregone value.

\subsubsection{Model Performance Benchmarking (RQ2)}

Achieving 89.26\% ROC-AUC for churn prediction and 92.25\% for claims frequency substantially exceeds industry benchmarks where performance above 0.80 is considered strong \citep{afriyie2024machine}. These results validate gradient boosting's superiority over traditional actuarial methods for insurance prediction tasks involving class imbalance and non-linear feature interactions, consistent with \citet{richman2023ai} and \citet{avanzi2023boosting}.

The claims severity model's $R^2 = 0.352$ (after leakage removal) warrants careful interpretation. While apparently modest, this performance represents honest predictive capability suitable for production deployment. The ``reduction'' from 0.645 with leakage features to 0.352 without them demonstrates the importance of rigorous feature engineering---models deployed with artificially inflated metrics fail in production when target-derived information is unavailable at prediction time.

\subsubsection{Premium Adequacy and Operational Deployment (RQ3, RQ4)}

The identification of 14.8\% of policies as systematically underpriced---where premiums fail to cover expected claims costs---represents a critical finding for portfolio profitability. This ``toxic revenue'' actively damages profitability with every renewal. Underpricing concentrates predictably: broker-sourced urban commercial vehicles show 22.3\% prevalence versus 7.8\% for agent-sourced rural passenger cars, suggesting organizational blind spots rather than random pricing errors.

The RAG deployment achieving 82\% production readiness with 24ms query latency validates \citet{gao2023retrieval}'s assertion that retrieval-augmented systems improve factual accuracy in knowledge-intensive domains. This performance enables real-time decision support previously infeasible with traditional analytics infrastructure, addressing RQ4's operational deployment question.

\subsection{Contributions to Literature}

This study makes several contributions to insurance analytics literature. First, the integrated multi-model framework addresses the gap identified by \citet{tharmarajan2024comparative} regarding isolated treatment of churn, claims, and value prediction. By connecting these models through CLV calculation, the framework enables holistic customer management.

Second, the rigorous treatment of data leakage in severity modeling provides methodological guidance for practitioners. The dramatic performance difference (0.645 vs. 0.352 $R^2$) after leakage removal highlights concerns about potentially inflated metrics in published research.

Third, the RAG operationalization demonstrates practical approaches for democratizing analytics access, extending \citet{lewis2020retrieval}'s foundational work to insurance-specific applications.

\subsection{Limitations}

Several limitations warrant acknowledgment. \textbf{Geographic Transferability:} The European dataset, while exhibiting universal insurance dynamics, may not capture Kenya-specific factors including fraud patterns, regulatory environment, and customer behavior. Direct validation on Kenyan data would strengthen emerging market applicability.

\textbf{Temporal Stability:} The 2015--2018 observation window captured relatively stable economic conditions. Model performance during recessions or fraud waves remains untested. Rolling validation across economic cycles would establish temporal robustness.

\textbf{Causal Inference:} While the pilot demonstrates intervention effectiveness (12.3\% churn reduction), absence of a randomized control group limits causal attribution. Observational data establishes predictive associations but does not prove specific causal mechanisms.

\textbf{Data Recency:} Telematics information (GPS tracking, driving behavior) increasingly available in modern insurance was absent from this dataset. Incorporating behavioral data could further improve prediction accuracy.

\subsection{Future Research Directions}

Four priority research trajectories emerge from this study:

\textbf{Temporal Stability Analysis:} Validation across 10+ years spanning multiple economic cycles to measure accuracy degradation and establish recalibration protocols.

\textbf{Causal Intervention Studies:} Randomized controlled trials with 15,000+ customers to quantify causal effects of specific retention interventions (discounts, proactive outreach, bundling offers).

\textbf{Multi-Line Integration:} Extension to household-level portfolios incorporating life, health, and property insurance to capture cross-product dynamics and wallet share optimization.

\textbf{Emerging Market Validation:} Direct replication using Kenyan portfolio data to validate transferability and calibrate for market-specific factors including fraud patterns and regulatory requirements.

% ============================================================================
% 6. CONCLUSION
% ============================================================================
\section{Conclusions}

This research developed and validated an integrated machine learning framework for automobile insurance customer analytics, addressing critical gaps in existing practice through four interdependent models achieving strong predictive performance: churn prediction at 89.26\% ROC-AUC, claims frequency at 92.25\% ROC-AUC, claims severity at $R^2 = 0.352$ (leakage-free), and portfolio valuation of \texteuro{}25.8 million with strategic four-quadrant segmentation.

Three empirical findings carry direct strategic implications. First, the ``lifecycle valley of death'' during policy years 1--3 concentrates 58\% elevated churn risk---retention investment should disproportionately target early-tenure customers where intervention ROI is maximized. Second, agent-sourced customers generate 2.5$\times$ higher ROI than broker channel despite lower premiums---strategic channel prioritization and broker pricing recalibration can recover millions in foregone value. Third, 14.8\% of policies are systematically underpriced---automated pricing adequacy monitoring can eliminate toxic revenue damaging profitability.

For practitioners, the framework enables immediate implementation: deploy churn scoring 60 days before renewal; implement differentiated strategies by segment (PROTECT: loyalty rewards; MANAGE: proactive outreach; DEVELOP: upsell campaigns; EXIT: strategic attrition); establish automated pricing adequacy alerts; and prioritize agent channel development.

The pilot deployment demonstrating 3,286\% first-year ROI validates that sophisticated customer analytics need not remain the exclusive domain of large, well-resourced insurers. In fragmented markets where acquisition costs reach 5--25$\times$ retention expense, knowing which customers will leave, when they'll make that decision, and which interventions work is not luxury analytics---it is survival intelligence.

\vspace{0.5em}
\begin{center}
\textbf{Interactive Application:} \url{https://automobilecustomerx.streamlit.app/}
\end{center}

% ============================================================================
% ACKNOWLEDGMENTS AND DECLARATIONS
% ============================================================================
\section*{Acknowledgments}
The author acknowledges Strathmore University for providing institutional support and computational resources necessary for this research.

\section*{Data Availability}
The dataset is publicly available through ICPSR repository. Processed datasets and analysis code are available upon reasonable request.

\section*{Conflict of Interest}
The author declares no conflicts of interest.

% ============================================================================
% REFERENCES
% ============================================================================
\bibliographystyle{cas-model2-names}
\bibliography{references}

\end{document}

