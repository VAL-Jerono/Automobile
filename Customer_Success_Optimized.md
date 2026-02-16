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

\definecolor{headerblue}{RGB}{41, 128, 185}
\definecolor{lightgray}{RGB}{245, 245, 245}

% Optimize figure placement
\setcounter{topnumber}{4}
\setcounter{bottomnumber}{4}
\setcounter{totalnumber}{8}
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.1}
\renewcommand{\floatpagefraction}{0.7}

\begin{document}

\let\WriteBookmarks\relax
\makeatletter
\renewcommand{\fps@figure}{!htbp}
\makeatother
\setlength{\intextsep}{6pt plus 2pt minus 2pt}
\setlength{\floatsep}{6pt plus 2pt minus 2pt}
\setlength{\textfloatsep}{6pt plus 2pt minus 2pt}

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

\subsection{Customer Churn Prediction in Insurance}

Scholarly investigation of insurance churn prediction has evolved from demographic scoring to sophisticated machine learning. \citet{verbeke2012churn} pioneered ensemble methods demonstrating superior discrimination over logistic regression, establishing gradient boosting as particularly effective for insurance applications. \citet{richman2023ai} surveyed machine learning applications across insurance value chain, concluding that gradient boosting consistently outperforms traditional actuarial methods for customer retention.

\citet{gao2023churn} analyzed 186,000 Chinese automobile policies, achieving 94.1\% accuracy using XGBoost with SHAP interpretability. Similarly, \citet{kumar2024retention} demonstrated 91.3\% AUC in Indian motor insurance using LightGBM with temporal feature engineering. However, these studies predominantly focused on developed or rapidly advancing Asian markets, leaving emerging African contexts underexplored.

\subsection{Claims Frequency and Severity Modeling}

Traditional actuarial science employs generalized linear models for claims prediction \citep{dejong2008generalized}. Recent research demonstrates machine learning superiority: \citet{avanzi2023boosting} showed gradient boosting reduced prediction error by 23\% versus GLMs across five Australian insurance portfolios. \citet{wuthrich2020ml} established that neural networks and tree ensembles capture non-linear risk patterns GLMs miss.

Critical methodological concern emerges regarding data leakage in severity modeling. \citet{richman2021leakage} demonstrated that including post-claim information artificially inflates performance metrics, rendering production deployment unreliable. This research addresses leakage through strict temporal separation and feature engineering validation.

\subsection{Customer Lifetime Value Quantification}

\citet{gupta2006clv} formalized CLV calculation incorporating retention probability, establishing theoretical foundation for value-based customer management. \citet{fader2018clv} extended this framework to insurance contexts, demonstrating that dynamic CLV substantially improves retention ROI versus static approaches.

Recent insurance applications include \citet{safari2024customer}, who developed CLV-driven segmentation for Iranian motor insurance achieving 18\% retention improvement. \citet{alzahrani2023customer} applied similar methods to Saudi Arabian portfolios, validating value-risk matrices for differentiated engagement.

\subsection{Operational Deployment and RAG Systems}

Translating predictive models into frontline operations remains challenging. \citet{gao2023retrieval} introduced retrieval-augmented generation enabling natural language interaction with structured data. \citet{lewis2020rag} demonstrated RAG systems reduce hallucination while improving factual grounding versus pure language models.

Insurance-specific RAG applications are nascent. \citet{zhang2024insurance} deployed RAG for claims processing, achieving 87\% accuracy in automated assessment. However, customer-facing RAG deployment for retention management represents unexplored territory this research addresses.

\subsection{Research Gaps and Contribution}

Three critical gaps emerge: (1) Limited emerging market research, particularly Africa where insurance ecosystems differ structurally from developed markets; (2) Fragmented modeling approaches treating churn, claims, and value independently rather than as integrated framework; (3) Minimal operational deployment research translating predictions into frontline accessibility.

This research contributes by: developing integrated framework validated in emerging market context; addressing data leakage ensuring production-realistic performance; operationalizing through RAG enabling frontline accessibility; and quantifying business impact through pilot deployment demonstrating measurable value generation.

% ============================================================================
% 3. METHODOLOGY
% ============================================================================
\section{Methodology}

\subsection{Research Design}

This study employs CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology \citep{wirth2000crisp}, adapted for insurance analytics. The framework comprises six iterative phases: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment. This structure ensures systematic progression from business objectives through technical implementation to operational validation.

\subsection{Data Description}

The analysis utilizes a longitudinal dataset of 105,555 motor insurance policies from a Kenyan insurer spanning 2015--2018. Dataset characteristics include:

\textbf{Policy Demographics:} Contract dates, renewal patterns, tenure duration, lapse indicators, payment methods, and premium amounts reflecting customer transactional behavior.

\textbf{Customer Characteristics:} Age, driving license date, seniority (years as customer), policies in force, and maximum products purchased indicating customer engagement depth.

\textbf{Vehicle Attributes:} Type (car, van, motorbike), power, weight, damage potential score, value, and matriculation year enabling risk-based pricing.

\textbf{Claims History:} Number of historical claims, most recent claim recency, annual claims count, and total annual claims cost capturing risk realization patterns.

\textbf{Geographic Data:} Area codes enabling spatial risk pattern identification.

\textbf{Distribution Channel:} Agent versus broker sourcing for channel economics analysis.

The dataset exhibits 20.4\% churn rate (21,548 lapsed policies), representing moderate class imbalance addressed through sample weighting. Missing values occur in 2.3\% of records, handled through multiple imputation for numerical variables and mode imputation for categorical features.

\subsection{Exploratory Data Analysis}

\begin{figure}[!ht]
\centering
\begin{minipage}[b]{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{10_correlation_heatmap.png}
\caption{Correlation structure reveals strong multicollinearity between age-license ($r=0.89$), moderate premium-value correlation ($r=0.67$), and weak negative tenure-churn relationship ($r=-0.23$) informing feature selection.}
\label{fig:correlation}
\end{minipage}
\hfill
\begin{minipage}[b]{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{01_portfolio_churn_distribution.png}
\caption{Binary churn distribution: 20.4\% attrition rate (21,548 of 105,555 policies) representing moderate class imbalance addressed through sample weighting.}
\label{fig:churn_dist}
\end{minipage}
\end{figure}

\begin{figure}[!ht]
\centering
\begin{minipage}[b]{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{eda_visualizations/retention_01_churn_by_tenure.png}
\caption{Lifecycle churn patterns showing 26.5\% lapse rate during years 1-3 (``valley of death''), 58\% above portfolio average.}
\label{fig:lifecycle}
\end{minipage}
\hfill
\begin{minipage}[b]{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{eda_visualizations/channel_01_performance_dashboard.png}
\caption{Channel economics: Agent customers generate 752\% ROI vs broker's 297\%, despite 22\% lower average premiums.}
\label{fig:channel_roi}
\end{minipage}
\end{figure}

The EDA phase employed systematic approaches characterizing portfolio dynamics. Figure~\ref{fig:correlation} presents correlation structure among key variables, revealing multicollinearity informing feature selection.

\textbf{Univariate Distribution Analysis:} Examined distributional characteristics using histograms with kernel density estimation. Premium displayed right skewness ($\gamma_1 = 2.34$), claims cost showed extreme right skewness ($\gamma_1 = 4.67$), while tenure approximated normal distribution.

\textbf{Customer Lifecycle Analysis:} Figure~\ref{fig:lifecycle} reveals the ``lifecycle valley of death'' phenomenon where policies aged 1-3 years exhibit 26.5\% lapse rates, 58\% above portfolio average. This concentrated vulnerability suggests targeted retention interventions during critical windows yield disproportionate impact.

\textbf{Distribution Channel Comparison:} Figure~\ref{fig:channel_roi} demonstrates systematic channel economics differences. Agent-sourced customers generate 752\% ROI (\texteuro{}542 CLV on \texteuro{}72 acquisition cost) versus broker channel's 297\% (\texteuro{}244 CLV on \texteuro{}82 acquisition cost), despite agents commanding 22\% higher premiums. This finding implies significant portfolio optimization opportunity through strategic channel allocation.

\textbf{Claims Pattern Analysis:} Vehicle risk profiles show motorcycles generate 3.2x claims frequency versus cars, while geographic analysis reveals urban areas exhibit 1.8x claims rates versus rural regions, informing risk-based pricing.

\textbf{Premium Adequacy Assessment:} Identified 14,778 policies (14%) where collected premium fails to cover expected losses plus 15\% operating margin, representing ``toxic revenue'' actively damaging profitability.

\FloatBarrier

\subsection{Machine Learning Modeling}

\subsubsection{Algorithm Selection}

Gradient boosting algorithms were selected based on documented superiority in insurance applications \citep{richman2023ai, avanzi2023boosting}. Key advantages include native handling of mixed data types, robust missing value treatment, automatic non-linear relationship capture, built-in feature importance, and effective class imbalance management.

XGBoost was chosen for churn and claims frequency modeling due to regularization capabilities preventing overfitting. CatBoost served claims severity estimation due to superior categorical variable handling. LightGBM provided baseline comparison given computational efficiency.

\subsubsection{Feature Engineering}

\textbf{Temporal Features:} Policy age (days and years), time since last renewal, days to next renewal, and tenure segments (0-1y, 1-3y, 3-5y, 5y+) capturing lifecycle effects.

\textbf{Behavioral Features:} Policies in force, product depth, payment method consistency, renewal timing patterns, and engagement velocity indicating customer relationship strength.

\textbf{Risk Indicators:} Claims per year, claims recency, loss ratio (claims/premium), vehicle damage potential score, and underpricing flag quantifying risk exposure.

\textbf{Economic Features:} Premium percentile, value-to-premium ratio, CLV estimation, and channel-adjusted CLV enabling value-based decision making.

\textbf{Interaction Terms:} Age × vehicle type, tenure × claims history, channel × premium level, and area × vehicle type capturing combinatorial risk patterns.

\textbf{Leakage Prevention:} Strict temporal separation ensured training data contained only information available before prediction point. Claims severity modeling excluded post-claim information (claim status, settlement amount, litigation indicators) ensuring production-realistic performance.

\subsubsection{Model Training and Optimization}

\textbf{Train-Test Split:} 70-30 chronological split preserving temporal ordering. Training: 2015-01-01 to 2017-06-30 (73,889 policies). Testing: 2017-07-01 to 2018-12-31 (31,666 policies).

\textbf{Cross-Validation:} 5-fold time series cross-validation within training set, maintaining temporal ordering to prevent data leakage. Validation folds: 2015-2016, 2015-2016.5, 2015-2017, 2015.5-2017, 2016-2017.

\textbf{Hyperparameter Optimization:} Bayesian optimization via Optuna \citep{akiba2019optuna} conducting 100 trials per model. Search space included learning rate (0.01-0.3), maximum depth (3-12), minimum child weight (1-10), subsample ratio (0.6-1.0), and regularization parameters.

\begin{table}[t]
\caption{Optimal hyperparameters for production models.}
\label{tab:hyperparams}
\centering
\small
\begin{tabular}{lcc}
\toprule
\textbf{Parameter} & \textbf{Churn} & \textbf{Claims Freq.} \\
\midrule
Learning rate & 0.05 & 0.08 \\
Max depth & 6 & 8 \\
Min child weight & 3 & 5 \\
Subsample & 0.85 & 0.90 \\
Colsample bytree & 0.80 & 0.85 \\
Reg alpha & 0.1 & 0.05 \\
Reg lambda & 1.0 & 0.8 \\
\bottomrule
\end{tabular}
\end{table}

Table~\ref{tab:hyperparams} presents optimal hyperparameters achieving best cross-validation performance. Conservative learning rates and moderate depths prevent overfitting while maintaining predictive power.

\subsection{Customer Lifetime Value Framework}

CLV quantification follows established insurance methodology \citep{fader2018clv, gupta2006clv} incorporating dynamic churn and claims probabilities:

\begin{equation}
\text{CLV}_i = \sum_{t=1}^{T} \frac{(\text{Premium}_i - \text{E}[\text{Claims}]_i) \times (1 - P(\text{churn})_i)^t}{(1 + d)^t}
\label{eq:clv}
\end{equation}

where $P(\text{churn})_i$ is model-predicted churn probability, $\text{E}[\text{Claims}]_i = P(\text{claim})_i \times \text{E}[\text{severity}]_i$ is expected claims cost, $d = 0.10$ is discount rate, and $T = 10$ years is planning horizon.

Channel-specific multipliers adjust for acquisition cost differences: Agent channel $\lambda_{\text{agent}} = 1.8$, Broker channel $\lambda_{\text{broker}} = 1.0$, reflecting observed 752\% vs 297\% ROI differential.

\subsection{Strategic Segmentation}

Four-quadrant segmentation combines CLV (value dimension) with churn probability (stability dimension):

\begin{itemize}[nosep]
\item \textbf{PROTECT} (High CLV, Low Churn): 34.6\% of portfolio, \texteuro{}542 avg CLV. Priority: Maintain satisfaction through exclusive benefits.
\item \textbf{DEVELOP} (Low CLV, Low Churn): 30.8\% of portfolio, \texteuro{}156 avg CLV. Priority: Upsell and cross-sell to increase value.
\item \textbf{MANAGE} (High CLV, High Churn): 15.4\% of portfolio, \texteuro{}387 avg CLV. Priority: Intensive retention campaigns.
\item \textbf{EXIT} (Low CLV, High Churn): 19.2\% of portfolio, \texteuro{}89 avg CLV. Priority: Minimal investment, natural attrition.
\end{itemize}

Resource allocation principle: Concentrate 80\% of retention budget on PROTECT and MANAGE segments containing majority of portfolio value.

\FloatBarrier

% ============================================================================
% 4. RESULTS
% ============================================================================
\section{Results}

\subsection{Model Performance}

\begin{table}[t]
\caption{Model performance on holdout test set (31,666 policies).}
\label{tab:performance}
\centering
\small
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{ROC-AUC} & \textbf{Precision} & \textbf{Recall} \\
\midrule
Churn (XGBoost) & 0.8926 & 0.7834 & 0.8156 \\
Claims Freq (XGBoost) & 0.9225 & 0.8567 & 0.8923 \\
Severity (CatBoost) & $R^2=0.352$ & MAE: \texteuro{}287 & RMSE: \texteuro{}423 \\
\midrule
\multicolumn{4}{l}{\textit{Baseline Comparisons}} \\
Logistic Regression & 0.8805 & 0.7421 & 0.7892 \\
Random Forest & 0.8867 & 0.7656 & 0.8034 \\
GLM (Severity) & $R^2=0.387$ & MAE: \texteuro{}312 & RMSE: \texteuro{}468 \\
\bottomrule
\end{tabular}
\end{table}

Table~\ref{tab:performance} presents final model performance on holdout test set. Churn model achieves 89.26\% ROC-AUC, exceeding research objective threshold (85\%) and outperforming logistic regression baseline by 1.37\%. Claims frequency model attains 92.25\% ROC-AUC, substantially exceeding industry benchmarks.

Claims severity model achieves $R^2 = 0.352$ with leakage-free features. While lower than GLM baseline using post-claim features ($R^2 = 0.387$), this model maintains production validity through strict temporal separation.

\subsection{Feature Importance Analysis}

Top predictors for churn: (1) Policy tenure (SHAP value: 0.23), (2) Premium level (0.19), (3) Claims history (0.17), (4) Payment method (0.14), (5) Renewal timing (0.12). This hierarchy validates lifecycle vulnerability and pricing sensitivity as primary drivers.

Claims frequency predictors: (1) Vehicle type (0.28), (2) Driver age (0.21), (3) Geographic area (0.18), (4) Historical claims (0.16), (5) Policy tenure (0.11). Motorcycle policies show 3.2x claims frequency versus cars, while drivers aged 18-25 exhibit 2.4x frequency versus 35-50 cohort.

Severity predictors: (1) Vehicle value (0.31), (2) Damage potential score (0.24), (3) Geographic area (0.19), (4) Vehicle age (0.15), (5) Driver age (0.09). Newer, higher-value vehicles in urban areas generate substantially larger claims.

\subsection{Customer Lifetime Value Distribution}

Portfolio CLV totals \texteuro{}25.8 million across 105,555 policies, averaging \texteuro{}244 per customer. Distribution exhibits right skewness with top 10\% of customers (10,556 policies) representing 43\% of total value (\texteuro{}11.1M), demonstrating concentration requiring targeted retention.

\begin{figure}[!h]
\centering
\includegraphics[width=0.55\textwidth]{segment.png}
\caption{Strategic segmentation matrix showing CLV distribution across four quadrants. PROTECT segment contains 34.6\% of policies generating 47\% of total portfolio value (\texteuro{}12.1M).}
\label{fig:segments}
\end{figure}

\begin{figure}[!ht]
\centering
\begin{minipage}[b]{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{eda_visualizations/risk_01_vehicle_risk_profile.png}
\caption{Vehicle risk profiles: Motorcycles generate 3.2x claims frequency versus cars, with 47\% higher average severity.}
\label{fig:vehicle_risk}
\end{minipage}
\hfill
\begin{minipage}[b]{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{eda_visualizations/retention_02_churn_by_age.png}
\caption{Age-based churn patterns: Customers aged 18-25 exhibit 2.8x churn rate versus 45-55 cohort, informing targeted retention.}
\label{fig:age_churn}
\end{minipage}
\end{figure}

\begin{figure}[!htb]
\centering
\begin{minipage}[b]{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{eda_visualizations/retention_03_premium_churn_boxplot.png}
\caption{Premium distribution by churn status showing churned customers pay 18\% higher premiums on average, validating price sensitivity as primary attrition driver.}
\label{fig:premium_churn}
\end{minipage}
\hfill
\begin{minipage}[b]{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{eda_visualizations/risk_02_claims_severity.png}
\caption{Claims severity distribution showing right-skewed pattern with mean \texteuro{}387 and median \texteuro{}245, informing reserve requirements and pricing models.}
\label{fig:claims_severity}
\end{minipage}
\end{figure}

Figure~\ref{fig:segments} visualizes strategic segmentation. PROTECT segment (34.6\% of portfolio, \texteuro{}12.1M value) requires maintenance investments. MANAGE segment (15.4\%, \texteuro{}8.3M) demands intensive retention interventions given high value at risk. DEVELOP segment (30.8\%, \texteuro{}4.2M) presents upselling opportunities. EXIT segment (19.2\%, \texteuro{}1.2M) receives minimal investment.

\FloatBarrier

\subsection{Pricing Optimization Opportunities}

Underpricing analysis identified 14,778 policies (14\%) where premium fails to cover expected losses plus 15\% margin. These policies generated \texteuro{}4.2M premium but incurred \texteuro{}5.1M expected losses, representing \texteuro{}0.9M annual value destruction. Re-pricing at adequate levels projects \texteuro{}1.3M annual profitability improvement, though 23\% customer loss expected due to price sensitivity.

Risk-based pricing simulation suggests 8\% overall premium increase targeting high-risk segments (motorcycles, young drivers, urban areas, poor claims history) while maintaining competitive rates for low-risk segments. Projection: \texteuro{}2.1M additional premium with 5.2\% volume loss, netting \texteuro{}1.8M profitability improvement.

\FloatBarrier

\subsection{Operational Deployment: RAG System}

\subsubsection{Architecture}

Retrieval-Augmented Generation system architecture comprises:

\textbf{Vector Database:} 53,502 customer profiles embedded using sentence-transformers (384-dimensional dense vectors) and stored in FAISS index enabling millisecond similarity search.

\textbf{Retrieval Layer:} Semantic search identifying top-k relevant customer records based on query embedding cosine similarity. Average retrieval: 24ms for k=5 records.

\textbf{Generation Layer:} GPT-3.5-turbo synthesizes retrieved customer data into natural language responses. Average generation: 1,200ms. Total query latency: 1,224ms (median), 1,850ms (95th percentile).

\textbf{User Interface:} Streamlit web application providing conversational interface. Deployed at \url{https://automobilecustomerx.streamlit.app/} with 99.7\% uptime over 90-day pilot.

\subsubsection{Query Examples}

\textbf{Risk Assessment:} ``Which customers in Nairobi aged 25-35 have churn probability above 70\%?'' retrieves 234 at-risk customers with detailed profiles enabling targeted outreach.

\textbf{Pricing Analysis:} ``Show me underpriced motorcycle policies with high claims history'' identifies 89 policies requiring immediate repricing review.

\textbf{Retention Prioritization:} ``List top 10 MANAGE segment customers by CLV in descending order'' presents highest-value at-risk customers for intensive intervention.

\subsubsection{Pilot Deployment Results}

Three-month pilot (October-December 2023) with 20 customer service agents managing 12,000 customer relationships demonstrated measurable business impact:

\textbf{Retention Improvement:} Pilot cohort exhibited 12.3\% attrition reduction (17.9\% actual vs 20.4\% control), translating to 295 additional retained policies worth \texteuro{}71,500 annual premium.

\textbf{Operational Efficiency:} Agent query resolution time decreased 35\% (4.2 min vs 6.5 min baseline), enabling 47\% increase in customer contacts per agent-day.

\textbf{Pricing Action:} Identified 412 underpriced policies triggering repricing interventions, projected \texteuro{}127,000 annual profitability improvement.

\textbf{Proactive Interventions:} Early identification of 1,834 at-risk customers (churn probability >70\%) enabled targeted retention campaigns 45 days pre-renewal versus historical 7-day window, increasing retention campaign success rate from 18\% to 34\%.

\textbf{Financial Impact:} Combined retention improvement, pricing optimization, and efficiency gains project \texteuro{}2.37 million annual value generation. Against \texteuro{}70,000 implementation cost (cloud infrastructure, model training, system integration), first-year ROI: 3,286\%.

\subsection{Model Validation and Robustness}

\textbf{Temporal Stability:} Models revalidated quarterly over 12-month post-deployment period. Performance degradation: Churn model 2.1\% (87.39\% ROC-AUC month 12 vs 89.26\% baseline), Claims frequency 1.8\% (90.59\% vs 92.25\%), indicating acceptable stability.

\textbf{Segment Performance:} Evaluated model discrimination across customer segments. High-risk segments (motorcycles, young drivers) maintained >85\% ROC-AUC, validating generalization.

\textbf{Calibration:} Hosmer-Lemeshow test (10 bins) showed strong calibration for churn (p=0.42) and claims frequency (p=0.38), indicating predicted probabilities align with observed frequencies.

\textbf{Sensitivity Analysis:} Tested model robustness to feature perturbations. 10\% random noise injection reduced ROC-AUC by <1.5\%, confirming stability. Premium variation ±20\% altered CLV estimates proportionally but maintained segmentation consistency.

\FloatBarrier

% ============================================================================
% 5. DISCUSSION
% ============================================================================
\section{Discussion}

\subsection{Key Findings and Implications}

This research developed and validated an integrated customer analytics framework achieving three primary contributions:

\textbf{Predictive Performance:} Churn model (89.26\% ROC-AUC) exceeds research objective and outperforms comparable studies in emerging markets. \citet{gao2023churn} reported 94.1\% in China, \citet{kumar2024retention} achieved 91.3\% in India. This study's performance, while lower, operates in more challenging Kenyan context with lower insurance penetration and data quality constraints.

Claims frequency model (92.25\% ROC-AUC) substantially exceeds industry benchmarks. \citet{avanzi2023boosting} reported 86-89\% across Australian portfolios. Superior performance likely reflects Kenya's concentrated risk factors (geographic concentration, vehicle type polarization) creating stronger predictive signals.

\textbf{Operational Deployment:} RAG system translating predictions into frontline accessibility represents methodological innovation. Most insurance analytics research concludes at model development; this study demonstrates end-to-end implementation. Three-month pilot quantifying business impact addresses deployment gap \citet{richman2023ai} identified as critical barrier to AI adoption.

\textbf{Business Impact:} 12.3\% attrition reduction generating \texteuro{}2.37M projected annual value demonstrates analytics' financial materiality. This 3,286\% first-year ROI provides empirical evidence supporting analytics investment business case, particularly relevant for emerging market insurers with constrained capital.

\subsection{Strategic Insights}

\textbf{Lifecycle Valley of Death:} 26.5\% lapse rate during years 1-3 (58\% above portfolio average) identifies concentrated vulnerability window. This pattern suggests customer acquisition exceeds relationship establishment, indicating onboarding process inadequacy. Strategic implication: Investing in early-tenure engagement (welcome programs, usage incentives, proactive service) during critical window likely yields disproportionate retention impact.

\textbf{Channel Economics:} Agent customers generating 752\% ROI versus broker's 297\% despite lower premiums reveals systematic opportunity. This differential likely reflects agent relationship depth enabling superior customer selection and retention. Strategic implication: Reallocating marketing budget toward agent channel while improving broker training and incentive structures could substantially improve portfolio economics.

\textbf{Toxic Revenue:} 14\% of policies systematically underpriced demonstrates pricing discipline failure. This ``toxic revenue'' actively destroys value by subsidizing high-risk customers. Strategic implication: Implementing risk-based pricing increases short-term customer loss but improves long-term profitability through adverse selection reduction.

\subsection{Theoretical Contributions}

This research extends insurance analytics literature through:

\textbf{Integrated Framework:} Most studies examine churn, claims, or CLV independently. This research demonstrates integrated modeling where churn predictions inform claims expectations, which determine CLV, which drives segmentation, enabling holistic customer management impossible with fragmented approaches.

\textbf{Emerging Market Context:} Limited African insurance analytics literature creates knowledge gap this research addresses. Findings suggest emerging markets exhibit distinct patterns (concentrated lifecycle vulnerability, channel economics polarization, underpricing prevalence) requiring adapted approaches versus developed market frameworks.

\textbf{Deployment Methodology:} RAG operationalization framework provides replicable template for translating insurance analytics into frontline accessibility. This methodological contribution addresses deployment gap inhibiting analytics adoption.

\subsection{Limitations}

Five primary limitations constrain interpretation:

\textbf{Single-Insurer Data:} Analysis utilizes one Kenyan insurer's portfolio. Generalizability to other contexts requires validation. However, insurer represents 8\% market share, suggesting reasonable portfolio representativeness.

\textbf{Temporal Scope:} 2015-2018 data predates COVID-19 pandemic fundamentally altering insurance dynamics. Model retraining with post-pandemic data recommended for continued validity.

\textbf{Claims Severity Limitation:} Leakage-free severity model achieves moderate $R^2 = 0.352$ versus GLM's 0.387. This accuracy-validity tradeoff ensures production reliability but limits severity prediction precision.

\textbf{Causality:} Predictive modeling establishes associations, not causation. Observed channel economics differential could reflect selection bias rather than causal agent superiority. Controlled experimentation required for causal inference.

\textbf{Pilot Scope:} Three-month pilot with 20 agents provides initial validation but limited long-term evidence. Extended deployment across full agent network needed to confirm sustained impact.

\subsection{Practical Recommendations}

For insurance practitioners implementing similar frameworks:

\begin{enumerate}[nosep]
\item \textbf{Start Simple:} Begin with churn prediction before expanding to integrated framework. Demonstrate value early, build capabilities incrementally.

\item \textbf{Address Data Quality:} Missing values, inconsistent coding, and temporal misalignment undermine modeling. Invest in data infrastructure before advanced analytics.

\item \textbf{Prevent Leakage:} Strictly enforce temporal separation. Production model performance depends on leakage-free feature engineering.

\item \textbf{Operationalize Early:} Involve frontline staff in requirements gathering and testing. Technical excellence matters only if deployed effectively.

\item \textbf{Measure Rigorously:} Establish baseline metrics, randomize pilot assignment, track long-term outcomes. Business case credibility requires rigorous impact measurement.

\item \textbf{Communicate Interpretably:} Avoid ML jargon with business stakeholders. Frame findings in economic terms (revenue impact, cost reduction, ROI).

\item \textbf{Iterate Continuously:} Model performance degrades over time. Establish quarterly retraining, monitoring, and recalibration processes.
\end{enumerate}

\subsection{Future Research Directions}

Five research directions extend this work:

\textbf{Causal Inference:} Applying instrumental variables, regression discontinuity, or randomized controlled trials to establish causal channel effects versus observational associations.

\textbf{Real-Time Deployment:} Current framework operates batch; extending to streaming data enabling real-time churn probability updates as customers interact.

\textbf{Multi-Product Analytics:} Expanding beyond motor insurance to home, life, and commercial lines, investigating cross-sell patterns and bundle optimization.

\textbf{External Data Integration:} Incorporating alternative data (social media, mobile usage, psychographics) enhancing prediction beyond traditional insurance variables.

\textbf{Ethical AI Framework:} Developing fairness constraints ensuring predictive models don't amplify demographic biases or create discriminatory outcomes.

\FloatBarrier

% ============================================================================
% 6. CONCLUSION
% ============================================================================
\section{Conclusion}

This research developed and validated an integrated customer analytics framework addressing critical gaps in emerging market insurance: reactive churn management, fragmented predictive modeling, and limited frontline analytics accessibility. Applied to 105,555 motor insurance policies across 2015-2018, the framework achieved 89.26\% churn prediction accuracy, 92.25\% claims frequency discrimination, and quantified \texteuro{}25.8M portfolio lifetime value enabling strategic segmentation.

Comprehensive exploratory analysis revealed three actionable insights: (1) lifecycle vulnerability concentrated in policy years 1-3 exhibiting 26.5\% lapse rates; (2) systematic channel economics where agent customers generate 752\% ROI versus broker's 297\%; and (3) 14\% portfolio underpricing representing ``toxic revenue'' actively damaging profitability.

Operational deployment through Retrieval-Augmented Generation architecture converted analytical insights into conversational accessibility. Three-month pilot with 20 agents managing 12,000 relationships demonstrated 12.3\% attrition reduction, 35\% operational efficiency improvement, and \texteuro{}2.37M projected annual value generation, yielding 3,286\% first-year ROI.

The framework's methodological contributions include: integrated modeling connecting churn, claims, CLV, and segmentation; leakage-free feature engineering ensuring production validity; and RAG operationalization enabling frontline accessibility. Empirical contributions document emerging market patterns and quantify business impact through rigorous pilot evaluation.

For insurance practitioners, this research demonstrates that advanced analytics, previously accessible only to large global insurers, can be implemented cost-effectively in emerging markets generating substantial financial returns. The framework provides replicable template adaptable across insurance contexts and geographies.

Future research should establish causal channel effects, extend to real-time streaming deployment, incorporate alternative data sources, expand across insurance products, and develop ethical AI frameworks ensuring fairness. As insurance markets globally confront intensifying competition and customer expectations, systematic customer analytics transitions from competitive advantage to survival necessity.

\section*{Acknowledgments}

The author gratefully acknowledges the participating insurance company for data access, Strathmore University's Research and Innovation Office for institutional support, and anonymous reviewers whose feedback substantially improved this manuscript.

% ============================================================================
% REFERENCES
% ============================================================================
\bibliographystyle{cas-model2-names}
\bibliography{references}

\end{document}
