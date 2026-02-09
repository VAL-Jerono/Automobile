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
\usepackage{caption}
\usepackage{subcaption}
\usepackage{multirow}
\usepackage{array}
\usepackage{longtable}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}

\usepackage{titlesec}
\titlespacing*{\subsection}{0pt}{1.5em}{1em}
\titlespacing*{\subsubsection}{0pt}{1.2em}{0.8em}

% Custom colors for tables
\definecolor{headerblue}{RGB}{41, 128, 185}
\definecolor{lightgray}{RGB}{245, 245, 245}

\begin{document}

\let\WriteBookmarks\relax
\def\floatpagepagefraction{1}
\def\textpagefraction{.001}

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
Customer retention represents a critical determinant of profitability in automobile insurance markets, where acquisition costs exceed retention investments by factors of five to twenty-five. This research develops and validates an integrated machine learning framework comprising four interdependent predictive models, applied to a longitudinal dataset of 105,555 motor insurance policies spanning 2015--2018.

The proposed framework synthesizes: (1) retention analytics achieving 89.26\% ROC-AUC discrimination accuracy for churn prediction using gradient boosting classification; (2) claims frequency modeling attaining 92.25\% ROC-AUC through ensemble methods; (3) claims severity estimation with $R^2 = 0.352$ using leakage-free feature engineering; and (4) customer lifetime value quantification totaling \texteuro{}25.8 million portfolio value with strategic segmentation into four actionable quadrants.

Exploratory analysis reveals three critical patterns: a ``lifecycle valley of death'' during policy years 1--3 exhibiting 26.5\% lapse rates (58\% above portfolio average); systematic channel economics where agent-sourced customers generate 752\% ROI versus broker channel's 297\%; and 14\% of contracts suffering systematic underpricing where premiums fail to cover expected losses.

The framework operationalizes insights through Retrieval-Augmented Generation (RAG) architecture, converting 53,502 customer profiles into conversationally accessible intelligence with 24ms median query latency. Three-month pilot deployment demonstrates 12.3\% attrition reduction, 35\% operational efficiency improvement, and \texteuro{}2.37 million projected annual value generation, yielding 3,386\% first-year return on \texteuro{}70,000 implementation investment.

\textbf{Interactive deployment:} \url{https://automobilecustomerx.streamlit.app/}
\end{abstract}

\begin{keywords}
Customer churn prediction \sep Insurance analytics \sep Gradient boosting \sep Customer lifetime value \sep Risk-based pricing \sep RAG systems \sep Emerging markets \sep XGBoost
\end{keywords}

\maketitle

% ============================================================================
% 1. INTRODUCTION
% ============================================================================
\section{Introduction}

\subsection{Background and Problem Context}

Policyholder retention has emerged as a fundamental driver of insurance profitability in competitive markets worldwide. Empirical research establishes that incremental improvements of 5\% in customer retention rates can amplify profits by 25\% to 95\%, demonstrating the asymmetric financial consequences of customer defection \citep{kumar2024retention}. This economic reality intensifies in emerging markets where insurance adoption remains structurally constrained and customer acquisition expenditures are disproportionately high relative to premium volumes.

Kenya's insurance penetration approximates 2.4\%, substantially below the global 7.0\% benchmark, implying that each lost policyholder represents not merely immediate premium erosion but forfeiture of considerable long-term customer lifetime value \citep{oecd2024insurance}. The challenge manifests acutely in automobile insurance, which constitutes Kenya's largest general insurance segment while experiencing severe structural pressures including market fragmentation, aggressive price competition, and elevated fraud levels \citep{wanjiru2023competitive, mwangi2024fraud}.

Within this environment, customer attrition materializes silently at renewal junctures, rendering retention simultaneously economically critical and operationally complex. Conventional customer management approaches have predominantly operated reactively: responding to churn after cancellation, to fraud after detection, and to profitability erosion after financial reporting reveals portfolio deterioration.

\subsection{Research Gap}

Analysis of contemporary insurance practice reveals four critical gaps:
\begin{enumerate}[label=(\roman*)]
    \item Insurers lack systematic frameworks for forecasting customer departure prior to renewal dates
    \item Claims prediction relies on traditional actuarial methodologies that fail to exploit behavioral data's predictive power
    \item Customer lifetime value calculations depend on static assumptions ignoring dynamic churn probabilities
    \item Customer segmentation remains predominantly demographic rather than behavioral
\end{enumerate}

\subsection{Research Objectives}

This study addresses these deficiencies through the following objectives:

\textbf{Primary Objective:} To develop and validate an integrated customer analytics framework combining machine learning prediction with operational deployment for automobile insurance in emerging markets.

\textbf{Specific Objectives:}
\begin{enumerate}
    \item To construct a churn prediction model achieving ROC-AUC $\geq 0.85$ for identifying at-risk policyholders
    \item To develop claims frequency and severity models enabling risk-based customer valuation
    \item To quantify customer lifetime value incorporating dynamic churn and claims probabilities
    \item To design strategic segmentation enabling differentiated retention interventions
    \item To operationalize analytics through natural language interfaces for frontline accessibility
\end{enumerate}

\subsection{Research Questions}

The study addresses the following research questions:
\begin{itemize}
    \item \textbf{RQ1:} What factors predict customer churn in automobile insurance portfolios?
    \item \textbf{RQ2:} How accurately can machine learning models forecast claims frequency and severity?
    \item \textbf{RQ3:} What is the economic value distribution across customer segments?
    \item \textbf{RQ4:} How can predictive analytics be operationalized for frontline decision-making?
\end{itemize}

% ============================================================================
% 2. LITERATURE REVIEW
% ============================================================================
\section{Literature Review}

\subsection{Customer Retention and Churn Prediction}

Customer defection constitutes a persistent profitability threat in competitive automobile insurance markets. Contemporary literature establishes that machine learning methodologies substantially surpass traditional statistical approaches in churn forecasting by capturing non-linear behavioral patterns and temporal dynamics.

\citet{zhang2023ensemble} demonstrated that ensemble learning techniques combining multiple gradient boosting algorithms attain superior predictive accuracy in insurance churn prediction, with discrimination accuracy exceeding 85\% in large-scale implementations. \citet{afriyie2024machine} showed through systematic comparison that tree-based models, particularly XGBoost and LightGBM, consistently outperform logistic regression and neural networks for insurance churn tasks due to native capability for handling class imbalance and capturing feature interactions automatically.

\citet{tharmarajan2024comparative} conducted comprehensive benchmarking across financial services datasets, confirming gradient boosting methods dominate churn prediction when appropriately tuned for imbalanced data. However, existing research predominantly treats churn prediction as an isolated task, rarely integrating churn risk with claims behavior or long-term customer value.

\subsection{Claims Frequency and Severity Modeling}

Precise prediction of claims frequency and severity represents fundamental requirements for underwriting discipline and profitability. Recent advances demonstrate that gradient boosting algorithms have emerged as the dominant modeling paradigm for claims analytics.

\citet{richman2023ai} provided comprehensive evidence that artificial intelligence approaches substantially outperform traditional generalized linear models for both claims frequency and severity prediction. \citet{avanzi2023boosting} demonstrated that tree-based boosting methods effectively handle zero-inflated characteristics of insurance claims data while maintaining interpretability through SHAP values. \citet{henckaerts2022boosting} showed that incorporating spatial and temporal features into gradient boosting frameworks significantly improves predictive accuracy.

\subsection{Customer Lifetime Value Estimation}

Customer Lifetime Value (CLV) has become a central metric guiding retention investment and portfolio optimization. \citet{chamberlain2024customer} demonstrated that machine learning regressors incorporating behavioral features significantly outperform linear models in predicting insurance lifetime value. \citet{liu2023predictive} showed that incorporating engagement metrics substantially improves CLV prediction accuracy. \citet{kumar2024retention} provided empirical evidence that value-based segmentation enables more effective retention strategies.

The CLV calculation follows actuarial principles:
\begin{equation}
    \text{CLV} = \sum_{t=1}^{T} \frac{(P_t - C_t - E_t) \cdot S_t}{(1 + r)^t} - A_0
    \label{eq:clv}
\end{equation}
where $P_t$ represents premium at time $t$, $C_t$ denotes expected claims cost, $E_t$ captures operating expenses, $S_t$ indicates survival probability, $r$ is the discount rate, and $A_0$ represents acquisition cost.

\subsection{Retrieval-Augmented Generation for Analytics}

Despite significant advances in predictive modeling, operational adoption remains constrained by interpretability barriers. Retrieval-Augmented Generation (RAG) has emerged as a promising solution by combining large language models with domain-specific knowledge retrieval.

\citet{gao2023retrieval} provided evidence that RAG architectures significantly reduce hallucinations and improve factual accuracy in knowledge-intensive domains. \citet{lewis2020retrieval} demonstrated that systems combining dense retrieval with generative models achieve state-of-the-art performance on question-answering tasks. \citet{zhao2024survey} showed that such systems outperform traditional chatbots in financial services applications.

% ============================================================================
% 3. METHODOLOGY
% ============================================================================
\section{Materials and Methods}

This study employs the Cross-Industry Standard Process for Data Mining (CRISP-DM) framework \citep{schroder2021crisp}, encompassing six phases: business understanding, data understanding, data preparation, modeling, evaluation, and deployment.

\subsection{Data Source and Characteristics}

\subsubsection{Dataset Origin and Scope}

Administrative records were obtained from ICPSR's open repository, representing a European insurance company's non-life motor vehicle insurance portfolio spanning November 1, 2015 to December 1, 2018. The dataset comprises $N = 105,555$ policy transactions with 30 variables across 53,502 unique policyholders.

\subsubsection{Variable Taxonomy}

Variables are categorized into five analytical domains as specified in Table~\ref{tab:variables}.

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

\subsection{Data Preprocessing}

\subsubsection{Missing Value Treatment}

Overall dataset completeness reached 97.39\%. Missing value mechanisms were classified and treated according to \citet{little2019statistical}:

\textbf{Structurally Missing:} Lapse date (85.5\% missing) represents right-censored observations in survival analysis terminology---policies not yet lapsed by observation window end.

\textbf{Missing Completely at Random (MCAR):} Vehicle length (9.8\%) and fuel type (1.7\%) exhibited random patterns confirmed through Little's MCAR test ($\chi^2 = 23.4$, $df = 18$, $p = 0.31$). K-Nearest Neighbors imputation ($k=5$) preserved multivariate relationships.

\textbf{Missing at Random (MAR):} Contract dates (57.2\%) exhibited patterns dependent on observed variables. Group-based deterministic imputation calculated contract start from renewal date minus seniority.

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
where $L_p$ and $U_p$ represent the $p$-th and $(100-p)$-th percentiles respectively.

Specific thresholds: Premium capped at 99th percentile (\texteuro{}1,200), vehicle value at 95th percentile (\texteuro{}45,000), vehicle power at 98th percentile (200 HP), and claim costs at 99th percentile (\texteuro{}8,500).

\subsubsection{Feature Engineering}

Systematic feature engineering created 98 composite variables from 30 raw features, organized into five modules:

\textbf{Module 1: Temporal Features}
\begin{equation}
    \text{Policy\_Age} = \frac{\text{Observation\_Date} - \text{Contract\_Start}}{365.25}
    \label{eq:policy_age}
\end{equation}

\begin{equation}
    \text{Tenure\_Loyalty\_Score} = \log(1 + \text{Seniority}) \times (1 + \text{Renewal\_Count})
    \label{eq:loyalty}
\end{equation}

\textbf{Module 2: Risk Indicators}
\begin{equation}
    \text{Composite\_Risk} = \alpha \cdot \text{Age\_Risk} + \beta \cdot \text{Vehicle\_Risk} + \gamma \cdot \text{Area\_Risk}
    \label{eq:composite_risk}
\end{equation}
where $\alpha$, $\beta$, $\gamma$ are domain-derived weights.

\textbf{Module 3: Financial Metrics}
\begin{equation}
    \text{Loss\_Ratio} = \frac{\text{Claims\_Cost}}{\text{Premium}}
    \label{eq:loss_ratio}
\end{equation}

\begin{equation}
    \text{Premium\_Adequacy} = \frac{\text{Premium}}{\mathbb{E}[\text{Claims}] + \text{Operating\_Cost}}
    \label{eq:premium_adequacy}
\end{equation}

\textbf{Module 4: Logarithmic Transforms}

Applied to highly skewed distributions (skewness $> 1.0$):
\begin{equation}
    x_{\text{log}} = \log(1 + x)
    \label{eq:log_transform}
\end{equation}

\textbf{Module 5: Interaction Features}

Channel-geography interactions, payment-tenure combinations, and claims history presence indicators.

\subsection{Exploratory Data Analysis Methodology}

The EDA phase employed seven analytical approaches to characterize portfolio dynamics:

\subsubsection{EDA-1: Univariate Distribution Analysis}

Examined distributional characteristics of all continuous variables using:
\begin{itemize}
    \item Histograms with kernel density estimation
    \item Box plots for outlier visualization
    \item Skewness ($\gamma_1$) and kurtosis ($\gamma_2$) statistics
\end{itemize}

\subsubsection{EDA-2: Bivariate Correlation Analysis}

Assessed relationships between predictors and targets using:
\begin{equation}
    r_{xy} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2 \sum_{i=1}^{n}(y_i - \bar{y})^2}}
    \label{eq:correlation}
\end{equation}

Correlation matrices visualized through heatmaps with hierarchical clustering.

\subsubsection{EDA-3: Customer Lifecycle Analysis}

Examined churn propensity across tenure cohorts to identify vulnerability windows:
\begin{equation}
    \text{Lapse\_Rate}_t = \frac{N_{\text{churned}, t}}{N_{\text{total}, t}} \times 100\%
    \label{eq:lapse_rate}
\end{equation}
where $t$ indexes policy tenure years.

\subsubsection{EDA-4: Distribution Channel Comparison}

Compared agent versus broker channels across multiple dimensions:
\begin{itemize}
    \item Customer lifetime value distributions
    \item Return on investment calculations
    \item Tenure and loss ratio comparisons
    \item Churn rate differentials
\end{itemize}

\subsubsection{EDA-5: Claims Pattern Analysis}

Characterized claims behavior by:
\begin{itemize}
    \item Vehicle type and geographic area
    \item Severity distributions (right-skewed, requiring log transformation)
    \item Frequency patterns across customer segments
\end{itemize}

\subsubsection{EDA-6: Premium Adequacy Assessment}

Identified underpriced policies where:
\begin{equation}
    \text{Underpriced} = \mathbb{I}\left[\text{Premium} < \mathbb{E}[\text{Claims}] \times (1 + \text{Loading})\right]
    \label{eq:underpriced}
\end{equation}

\subsubsection{EDA-7: Customer Segmentation Profiling}

Applied value-risk matrix segmentation:
\begin{equation}
    \text{Segment} = f(\text{CLV\_Quartile}, \text{Risk\_Quartile})
    \label{eq:segmentation}
\end{equation}
producing four quadrants: PROTECT, DEVELOP, MANAGE, EXIT.

\subsection{Machine Learning Architecture}

\subsubsection{Algorithm Selection Rationale}

Gradient boosting algorithms were selected based on documented superiority in insurance applications \citep{richman2023ai, avanzi2023boosting}. Key advantages include:
\begin{itemize}
    \item Native handling of mixed data types
    \item Robust treatment of missing values via surrogate splits
    \item Automatic capture of non-linear relationships
    \item Built-in feature importance metrics
    \item Effective class imbalance management
\end{itemize}

\subsubsection{Model 1: Churn Prediction}

Gradient Boosting Classifier configured with:
\begin{itemize}
    \item $n_{\text{estimators}} = 100$ sequential trees
    \item Learning rate $\eta = 0.1$
    \item Maximum depth $d_{\max} = 5$
    \item Class weight $w = 3.90$
\end{itemize}

The objective function minimizes binary cross-entropy:
\begin{equation}
    \mathcal{L}_{\text{churn}} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\right]
    \label{eq:bce_loss}
\end{equation}

\subsubsection{Model 2: Claims Frequency}

Identical architecture with class weight $w = 4.37$ reflecting higher imbalance.

\subsubsection{Model 3: Claims Severity}

Gradient Boosting Regressor trained on claimants only ($N = 19,646$). Target variable log-transformed to address right skewness:
\begin{equation}
    \hat{y}_{\text{severity}} = \exp\left(\hat{y}_{\log}\right) - 1
    \label{eq:severity_backtransform}
\end{equation}

\textbf{Data Leakage Prevention:} Severity-derived features excluded from predictor set:
\begin{itemize}
    \item \texttt{Is\_severe\_claim} (target-derived binary)
    \item \texttt{Severity\_log} (log-transformed target)
    \item \texttt{Loss\_ratio} (calculated using target)
    \item \texttt{Is\_unprofitable} (loss ratio threshold)
\end{itemize}

\subsubsection{Model 4: Customer Lifetime Value}

Probabilistic integration of Models 1--3:
\begin{equation}
    \text{CLV}_i = \sum_{t=1}^{10} \left[(P_{i,t} \times 0.75 - \hat{C}_{i,t} - E_t) \times \hat{S}_{i,t} \times (1.05)^{-t}\right] - A_i
    \label{eq:clv_full}
\end{equation}

where:
\begin{align}
    \hat{S}_{i,t} &= \prod_{j=1}^{t}(1 - \hat{p}_{\text{churn},i}) \label{eq:survival} \\
    \hat{C}_{i,t} &= \hat{p}_{\text{freq},i} \times \hat{y}_{\text{severity},i} \label{eq:expected_claims}
\end{align}

\subsubsection{Model 5: Strategic Segmentation}

Rule-based classification using CLV and claims risk quartiles:

\begin{table}[H]
\centering
\caption{Strategic segmentation matrix combining customer lifetime value quartiles with claims risk quartiles to create four actionable customer groups with differentiated management strategies.}
\label{tab:segmentation}
\begin{tabular}{@{}lcc@{}}
\toprule
& \textbf{Low Claims Risk} & \textbf{High Claims Risk} \\
\midrule
\textbf{High CLV} & PROTECT & MANAGE \\
\textbf{Low CLV} & DEVELOP & EXIT \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Validation Methodology}

\subsubsection{Temporal Train-Test Split}

Temporal validation prevents information leakage across time boundaries:
\begin{itemize}
    \item \textbf{Training Set:} 2015--2017 data ($N = 84,444$, 80\%)
    \item \textbf{Test Set:} 2018 data ($N = 21,111$, 20\%)
\end{itemize}

\subsubsection{Hyperparameter Optimization}

Bayesian optimization via Optuna with Tree-structured Parzen Estimator (TPE):
\begin{itemize}
    \item Search space: $\eta \in [0.01, 0.30]$, $d_{\max} \in [3, 10]$, subsample $\in [0.6, 1.0]$
    \item Trials: 50 (classification), 80 (regression)
    \item Objective: Maximize ROC-AUC (classification), $R^2$ (regression)
    \item Validation: Stratified 5-fold cross-validation
\end{itemize}

\subsection{Evaluation Metrics}

\subsubsection{Classification Metrics}

\textbf{ROC-AUC:} Area under Receiver Operating Characteristic curve, interpretable as probability that a randomly selected positive case receives higher predicted probability than a randomly selected negative case.

\textbf{Precision-Recall AUC:} Complementary metric for imbalanced datasets with baseline equal to minority class prevalence.

\subsubsection{Regression Metrics}

\begin{equation}
    R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
    \label{eq:r_squared}
\end{equation}

\begin{equation}
    \text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
    \label{eq:mae}
\end{equation}

\subsection{Deployment Architecture}

\subsubsection{RAG System Implementation}

The Retrieval-Augmented Generation system comprises:
\begin{enumerate}
    \item \textbf{Document Processing:} Policy records converted to natural language descriptions
    \item \textbf{Embedding Generation:} SentenceTransformer (all-MiniLM-L6-v2) producing 384-dimensional vectors
    \item \textbf{Vector Database:} FAISS IndexFlatL2 for exact nearest neighbor search
    \item \textbf{Query Interface:} Natural language to structured retrieval
\end{enumerate}

% ============================================================================
% 4. RESULTS
% ============================================================================
\section{Results}

\subsection{Data Collection and Quality Assessment}

The final analytical dataset comprised 105,555 policy records representing 53,502 unique policyholders across a 37-month observation window. Data quality metrics are summarized in Table~\ref{tab:data_quality}.

\begin{table}[H]
\centering
\caption{Data quality assessment summary showing completeness metrics, imputation actions, and final dataset characteristics after preprocessing pipeline execution.}
\label{tab:data_quality}
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Metric} & \textbf{Value} & \textbf{Percentage} \\
\midrule
Total Records & 105,555 & 100.0\% \\
Unique Policies & 53,502 & 50.7\% \\
Complete Records & 102,799 & 97.39\% \\
Records with Missing Values & 2,756 & 2.61\% \\
Duplicate Records Removed & 0 & 0.0\% \\
Outliers Winsorized & 8,641 & 8.19\% \\
\midrule
Final Training Set & 84,444 & 80.0\% \\
Final Test Set & 21,111 & 20.0\% \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Exploratory Data Analysis Findings}

\subsubsection{EDA-1: Distribution Characteristics}

Continuous variables exhibited varying distributional properties requiring transformation. Premium displayed right skewness ($\gamma_1 = 2.34$), claims cost showed extreme right skewness ($\gamma_1 = 4.67$), while tenure approximated normal distribution ($\gamma_1 = 0.89$).

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{distribution_analysis.png}
\caption{Univariate distribution analysis of key continuous variables. Premium and claims cost exhibit pronounced right skewness necessitating logarithmic transformation for regression modeling. Tenure shows approximate normality with slight positive skew. Box plots (right panels) highlight outlier prevalence in monetary variables.}
\label{fig:distributions}
\end{figure}

\subsubsection{EDA-2: Correlation Structure}

Correlation analysis revealed strong multicollinearity between driver age and license years ($r = 0.89$), moderate correlation between premium and vehicle value ($r = 0.67$), and weak correlation between tenure and churn ($r = -0.23$).

\begin{figure}[H]
\centering
\includegraphics[width=0.80\textwidth]{correlation_heatmap.png}
\caption{Correlation matrix heatmap with hierarchical clustering revealing variable relationship structure. Strong positive correlation between driver age and license years suggests potential redundancy. Negative correlations between tenure and churn validate theoretical expectations. Claims history shows moderate positive correlation with claims frequency target.}
\label{fig:correlation}
\end{figure}

\subsubsection{EDA-3: Customer Lifecycle Vulnerability}

Lapse rate analysis across tenure cohorts revealed a pronounced vulnerability window during policy years 1--3, as illustrated in Figure~\ref{fig:lifecycle}.

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{Lifecycle.png}
\caption{Customer lifecycle vulnerability curve showing lapse rates by tenure year. The ``valley of death'' during years 1--3 exhibits 26.5\% average lapse rate, representing 58\% elevation above portfolio average (20.4\%) and 59\% higher than mature relationships exceeding 10 years (16.7\%). Error bars indicate 95\% confidence intervals.}
\label{fig:lifecycle}
\end{figure}

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

\subsubsection{EDA-4: Distribution Channel Economics}

Agent-sourced policies demonstrated systematic advantages across multiple performance dimensions compared to broker-sourced policies.

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{ROI and Channel.png}
\caption{Distribution channel comparative economics across five dimensions. Agent channel demonstrates 2.5$\times$ higher ROI (752\% vs. 297\%), 60\% longer tenure (8.23 vs. 4.84 years), 14\% lower loss ratios (44.3\% vs. 53.4\%), and 21\% reduced churn propensity (16.2\% vs. 20.5\%). The \texteuro{}483 CLV gap persists despite broker customers paying 8.5\% higher premiums.}
\label{fig:channel}
\end{figure}

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

\subsubsection{EDA-5: Claims Pattern Analysis}

Claims frequency and severity exhibited systematic variation by vehicle type and geographic area.

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{claims_area_vehicle.png}
\caption{Claims frequency by vehicle type and geographic area. Urban policies consistently exhibit higher claims rates across all vehicle categories, with vans showing the largest urban-rural differential (23.4\% vs. 14.2\%). Agricultural vehicles demonstrate lowest claims frequency regardless of area, reflecting lower traffic exposure.}
\label{fig:claims_patterns}
\end{figure}

\begin{table}[H]
\centering
\caption{Claims frequency decomposition by vehicle type and geographic area showing systematic risk patterns informing pricing and underwriting decisions.}
\label{tab:claims}
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{Vehicle Type} & \textbf{N} & \textbf{Rural} & \textbf{Urban} & \textbf{Overall} & \textbf{Urban Premium} \\
\midrule
Passenger Car & 83,725 & 16.2\% & 22.8\% & 18.6\% & $+40.7\%$ \\
Van & 13,938 & 14.2\% & 23.4\% & 17.1\% & $+64.8\%$ \\
Motorbike & 5,067 & 19.8\% & 28.3\% & 22.4\% & $+42.9\%$ \\
Agricultural & 2,825 & 8.4\% & 12.1\% & 9.2\% & $+44.0\%$ \\
\midrule
\textbf{All Types} & \textbf{105,555} & \textbf{15.7\%} & \textbf{22.9\%} & \textbf{18.6\%} & $+45.9\%$ \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{EDA-6: Premium Adequacy Assessment}

Pure premium analysis identified 14.8\% of policies as systematically underpriced, where collected premium fails to cover expected claims costs.

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{pricing_optimization.png}
\caption{Premium adequacy analysis by segment. Underpricing concentrates in broker-sourced urban commercial vehicles (22.3\% prevalence) versus agent-sourced rural passenger cars (7.8\%). Color intensity indicates underpricing severity as percentage of expected loss not covered by premium.}
\label{fig:pricing}
\end{figure}

\subsubsection{EDA-7: Customer Segmentation Distribution}

The value-risk matrix classified the portfolio into four strategic segments.

\begin{figure}[H]
\centering
\includegraphics[width=0.80\textwidth]{segment.png}
\caption{Customer journey segmentation matrix showing portfolio distribution and mean CLV by quadrant. PROTECT segment (34.6\%, mean CLV \texteuro{}542) comprises high-value low-risk customers. DEVELOP (30.8\%, \texteuro{}156) represents growth opportunities. MANAGE (15.4\%, \texteuro{}387) requires proactive risk management. EXIT (19.2\%, \texteuro{}89) contains candidates for strategic attrition.}
\label{fig:segments}
\end{figure}

\begin{table}[H]
\centering
\caption{Strategic segment characteristics including portfolio share, mean CLV, churn risk, claims frequency, and recommended management strategy.}
\label{tab:segments}
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{Segment} & \textbf{Share} & \textbf{Mean CLV} & \textbf{Churn Risk} & \textbf{Claims Freq.} & \textbf{Strategy} \\
\midrule
PROTECT & 34.6\% & \texteuro{}542 & 12.3\% & 11.2\% & Loyalty \& retention \\
DEVELOP & 30.8\% & \texteuro{}156 & 15.8\% & 13.4\% & Upsell \& engagement \\
MANAGE & 15.4\% & \texteuro{}387 & 28.7\% & 31.2\% & Risk mitigation \\
EXIT & 19.2\% & \texteuro{}89 & 34.5\% & 29.8\% & Strategic attrition \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Predictive Model Performance}

\subsubsection{Model 1: Churn Prediction}

The optimized churn model achieved ROC-AUC of 0.8926 on the temporal holdout test set, exceeding the target threshold of 0.85.

\begin{table}[H]
\centering
\caption{Churn prediction model performance comparison showing baseline versus optimized configurations. Bayesian optimization achieved 1.4\% improvement in discrimination accuracy.}
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

\subsubsection{Model 2: Claims Frequency}

Claims frequency prediction achieved ROC-AUC of 0.9225, demonstrating strong discrimination ability.

\begin{table}[H]
\centering
\caption{Claims frequency model performance showing near-optimal baseline configuration with marginal optimization gains.}
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

\subsubsection{Model 3: Claims Severity}

After removing leakage features, the severity model achieved $R^2 = 0.352$, representing realistic predictive capability for production deployment.

\begin{table}[H]
\centering
\caption{Claims severity model performance with leakage features removed. The ``reduction'' from 0.645 to 0.352 $R^2$ represents correction to honest metrics, not model degradation.}
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

\subsubsection{Model 4: Customer Lifetime Value}

CLV calculations quantified total portfolio value at \texteuro{}25.8 million with substantial variation across segments.

\begin{table}[H]
\centering
\caption{Customer lifetime value distribution showing portfolio-level and segment-level value quantification.}
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

\subsubsection{Feature Importance Analysis}

Feature importance analysis revealed historical claims rate as the dominant predictor across all models.

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{feature_importance.png}
\caption{Top 10 feature importances for churn and claims frequency models. Historical claims rate dominates both models, followed by tenure, premium, and vehicle characteristics. Feature importance calculated using mean decrease in impurity across all trees.}
\label{fig:importance}
\end{figure}

\subsection{RAG System Performance}

The RAG deployment achieved 82\% production readiness with sub-second query response times.

\begin{table}[H]
\centering
\caption{RAG system performance metrics demonstrating production readiness for natural language query interface.}
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

Three-month pilot deployment with 20 agents managing 12,000 customer relationships demonstrated measurable business impact.

\begin{table}[H]
\centering
\caption{Pilot deployment results demonstrating operational impact across retention, efficiency, and financial dimensions.}
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

The finding aligns with \citet{kumar2024retention}'s observation that relationship commitment follows a non-linear trajectory, but extends this insight by quantifying the specific vulnerability window in automobile insurance. For practitioners, this implies that retention investment should disproportionately target the 1--3 year cohort, where intervention ROI is maximized. Preventing early-stage defection converts high-risk customers into stable, long-term relationships that contribute disproportionately to portfolio value.

\subsubsection{Distribution Channel Economics (RQ1, RQ3)}

The 2.5$\times$ ROI differential between agent (752\%) and broker (297\%) channels represents one of the study's most strategically significant findings. This gap persists despite broker customers paying 8.5\% higher premiums, indicating systematic underpricing relative to actual risk profiles in the broker channel.

Three compounding factors drive agent channel superiority: 60\% longer tenure (8.23 vs. 4.84 years), 14\% lower loss ratios (44.3\% vs. 53.4\%), and 21\% reduced churn propensity. These findings contradict conventional assumptions that higher premiums indicate adequate pricing. Instead, broker channel's elevated premiums fail to compensate for substantially higher claims costs and shorter customer lifespans.

This finding carries direct strategic implications. Insurers should prioritize agent channel development over broker expansion, while broker channel pricing requires recalibration to adequately reflect elevated risk profiles. The \texteuro{}483 per-customer lifetime value difference, extrapolated across a typical portfolio, represents millions of euros in foregone value.

\subsubsection{Model Performance Benchmarking (RQ2)}

Achieving 89.26\% ROC-AUC for churn prediction and 92.25\% for claims frequency substantially exceeds industry benchmarks where performance above 0.80 is considered strong \citep{afriyie2024machine}. These results validate gradient boosting's superiority over traditional actuarial methods for insurance prediction tasks involving class imbalance and non-linear feature interactions.

The claims severity model's $R^2 = 0.352$ (after leakage removal) warrants careful interpretation. While apparently modest, this performance represents honest predictive capability suitable for production deployment. The ``reduction'' from 0.645 with leakage features to 0.352 without them demonstrates the importance of rigorous feature engineering. Models deployed with artificially inflated metrics fail in production when target-derived information is unavailable at prediction time.

\subsubsection{Premium Adequacy Gaps (RQ3)}

The identification of 14.8\% of policies as systematically underpriced---where premiums fail to cover expected claims costs---represents a critical finding for portfolio profitability. This ``toxic revenue'' actively damages profitability with every renewal.

Underpricing concentrates predictably: broker-sourced urban commercial vehicles show 22.3\% prevalence versus 7.8\% for agent-sourced rural passenger cars. This systematic pattern suggests organizational blind spots rather than random pricing errors. Implementing automated pricing adequacy flags enables proactive identification of underpriced renewals before contracts roll over.

\subsection{Contextualizing Within Existing Literature}

The study's findings both confirm and extend existing research. The superiority of gradient boosting for insurance applications aligns with \citet{richman2023ai} and \citet{avanzi2023boosting}, while the integrated multi-model framework addresses the gap identified by \citet{tharmarajan2024comparative} regarding isolated treatment of churn, claims, and value prediction.

The RAG deployment achieving 82\% production readiness confirms \citet{gao2023retrieval}'s assertion that retrieval-augmented systems improve factual accuracy in knowledge-intensive domains. The 24ms query latency enables real-time decision support previously infeasible with traditional analytics infrastructure.

\subsection{Unexpected Findings}

Hyperparameter optimization yielded unexpectedly modest improvements. For churn prediction, optimization achieved only 1.4\% ROC-AUC improvement (0.8805 to 0.8926), while claims frequency showed 0.15\% gain. This suggests that domain-informed baseline configurations approached optimal performance, and further gains require feature engineering rather than algorithmic tuning.

The severity model's dramatic performance ``decline'' after leakage removal (0.645 to 0.352 $R^2$) was expected but highlights a broader concern: published insurance analytics research may inadvertently report inflated metrics when target-derived features contaminate predictor sets.

\subsection{Limitations}

Several limitations warrant acknowledgment:

\textbf{Geographic Transferability:} The European dataset, while exhibiting universal insurance dynamics, may not capture Kenya-specific factors including fraud patterns, regulatory environment, and customer behavior. Direct validation on Kenyan data would strengthen emerging market applicability.

\textbf{Temporal Stability:} The 2015--2018 observation window captured relatively stable economic conditions. Model performance during recessions or fraud waves remains untested. Rolling validation across economic cycles would establish temporal robustness.

\textbf{Causal Inference:} While the pilot demonstrates intervention effectiveness (12.3\% churn reduction), absence of a randomized control group limits causal attribution. Observational data establishes predictive associations but does not prove specific causal mechanisms.

\textbf{Data Recency:} Telematics information (GPS tracking, driving behavior) increasingly available in modern insurance was absent from this dataset. Incorporating behavioral data could further improve prediction accuracy.

\subsection{Future Research Directions}

Five research trajectories emerge from this study:

\begin{enumerate}
    \item \textbf{Temporal Stability Analysis:} Validation across 10+ years spanning multiple economic cycles to measure accuracy degradation and establish recalibration protocols.
    
    \item \textbf{Causal Intervention Studies:} Randomized controlled trials with 15,000+ customers to quantify causal effects of specific retention interventions (discounts, check-ins, bundling).
    
    \item \textbf{Multi-Line Integration:} Extension to household-level portfolios incorporating life, health, and property insurance to capture cross-product dynamics and wallet share optimization.
    
    \item \textbf{Advanced RAG Architectures:} Multi-modal retrieval combining structured queries with unstructured documents (policy contracts, claim notes, customer communications).
    
    \item \textbf{Emerging Market Validation:} Direct replication using Kenyan portfolio data to validate transferability and calibrate for market-specific factors.
\end{enumerate}

% ============================================================================
% 6. CONCLUSIONS
% ============================================================================
\section{Conclusions}

\subsection{Summary of Contributions}

This research developed and validated an integrated machine learning framework for automobile insurance customer analytics, addressing critical gaps in existing practice. The framework comprises four interdependent models:

\begin{enumerate}
    \item \textbf{Churn Prediction:} 89.26\% ROC-AUC discrimination accuracy identifying at-risk policyholders for targeted retention intervention.
    
    \item \textbf{Claims Analytics:} 92.25\% ROC-AUC for frequency prediction and $R^2 = 0.352$ for severity estimation using leakage-free feature engineering.
    
    \item \textbf{Lifetime Value Quantification:} \texteuro{}25.8 million portfolio valuation with \texteuro{}483 systematic value differential between distribution channels.
    
    \item \textbf{Strategic Segmentation:} Four-quadrant classification enabling differentiated customer management aligned with value-risk trajectories.
\end{enumerate}

\subsection{Key Findings and Implications}

Three empirical findings carry direct strategic implications:

\textbf{Finding 1:} The ``lifecycle valley of death'' during policy years 1--3 concentrates 58\% elevated churn risk. \textit{Implication:} Retention investment should disproportionately target early-tenure customers where intervention ROI is maximized.

\textbf{Finding 2:} Agent-sourced customers generate 2.5$\times$ higher ROI than broker channel despite lower premiums. \textit{Implication:} Strategic channel prioritization and broker pricing recalibration can recover millions in foregone value.

\textbf{Finding 3:} 14.8\% of policies are systematically underpriced below expected loss costs. \textit{Implication:} Automated pricing adequacy monitoring can eliminate ``toxic revenue'' that damages profitability with every renewal.

\subsection{Practical Recommendations}

For insurance practitioners, the framework enables immediate implementation:

\begin{itemize}
    \item Deploy churn prediction scoring for all renewals 60 days before expiration
    \item Implement differentiated retention strategies by segment (PROTECT: loyalty rewards; MANAGE: proactive outreach; DEVELOP: upsell campaigns; EXIT: strategic attrition)
    \item Establish automated pricing adequacy alerts flagging underpriced renewals
    \item Prioritize agent channel development with target 70\% portfolio share
    \item Recalibrate broker channel pricing with 10\% increase reflecting actual risk profiles
\end{itemize}

\subsection{Concluding Remarks}

The pilot deployment demonstrating 3,286\% first-year ROI validates that sophisticated customer analytics need not remain the exclusive domain of large, well-resourced insurers. Cloud infrastructure has eliminated capital barriers. The methodological foundations established here provide a roadmap for any insurer serious about transforming from reactive firefighting to proactive customer success.

In fragmented markets where customer acquisition costs reach 5--25$\times$ retention expense, knowing which customers will leave, when they'll make that decision, and which interventions actually work is not luxury analytics---it is survival intelligence. The silent crisis of customer churn has a data-driven solution. This research provides the roadmap.

\vspace{1em}
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
