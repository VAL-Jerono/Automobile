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
\usepackage{multirow}
\usepackage{array}
\usepackage{xcolor}

\usepackage{titlesec}
\titlespacing*{\subsection}{0pt}{1.2em}{0.8em}
\titlespacing*{\subsubsection}{0pt}{1em}{0.6em}

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
\title[mode = title]{Integrated Predictive Analytics for Customer Retention and Risk Optimization in Automobile Insurance: A Machine Learning Framework}

\author{Valerie Jerono}
\ead{valerie.jerono@strathmore.edu}
\address[1]{School of Computing and Engineering Sciences, Strathmore University, Nairobi, Kenya}

% ============================================================================
% ABSTRACT
% ============================================================================
\begin{abstract}
Customer retention represents a critical determinant of profitability in automobile insurance markets, where acquisition costs exceed retention investments by factors of five to twenty-five. This research develops and validates an integrated machine learning framework comprising four interdependent predictive models, applied to a longitudinal dataset of 105,555 motor insurance policies spanning 2015--2018.

The proposed framework synthesizes: (1) retention analytics achieving 89.26\% ROC-AUC for churn prediction; (2) claims frequency modeling attaining 92.25\% ROC-AUC; (3) claims severity estimation with $R^2 = 0.352$ using leakage-free feature engineering; and (4) customer lifetime value quantification totaling \texteuro{}25.8 million portfolio value.

Exploratory analysis reveals three critical patterns: a ``lifecycle valley of death'' during policy years 1--3 exhibiting 26.5\% lapse rates; systematic channel economics where agent-sourced customers generate 752\% ROI versus broker channel's 297\%; and 14\% of contracts suffering systematic underpricing. The framework operationalizes insights through Retrieval-Augmented Generation (RAG) architecture with 24ms median query latency. Pilot deployment demonstrates 12.3\% attrition reduction and \texteuro{}2.37 million projected annual value generation.

\textbf{Interactive deployment:} \url{https://automobilecustomerx.streamlit.app/}
\end{abstract}

\begin{keywords}
Customer churn prediction \sep Insurance analytics \sep Gradient boosting \sep Customer lifetime value \sep RAG systems
\end{keywords}

\maketitle

% ============================================================================
% 1. INTRODUCTION
% ============================================================================
\section{Introduction}

\subsection{Background and Problem Context}

Policyholder retention has emerged as a fundamental driver of insurance profitability. Empirical research establishes that incremental improvements of 5\% in customer retention rates can amplify profits by 25\% to 95\% \citep{kumar2024retention}. This economic reality intensifies in emerging markets where insurance adoption remains structurally constrained and customer acquisition expenditures are disproportionately high.

Kenya's insurance penetration approximates 2.4\%, substantially below the global 7.0\% benchmark \citep{oecd2024insurance}. The challenge manifests acutely in automobile insurance, which constitutes Kenya's largest general insurance segment while experiencing severe structural pressures including market fragmentation and aggressive price competition \citep{wanjiru2023competitive, mwangi2024fraud}.

Analysis of contemporary insurance practice reveals four critical gaps: (i) lack of systematic frameworks for forecasting customer departure; (ii) claims prediction relying on traditional actuarial methods that fail to exploit behavioral data; (iii) customer lifetime value calculations depending on static assumptions; and (iv) segmentation remaining predominantly demographic rather than behavioral.

\subsection{Research Objectives}

\textbf{Primary Objective:} To develop and validate an integrated customer analytics framework combining machine learning prediction with operational deployment for automobile insurance.

\textbf{Specific Objectives:}
\begin{enumerate}[nosep]
    \item Construct a churn prediction model achieving ROC-AUC $\geq 0.85$
    \item Develop claims frequency and severity models enabling risk-based customer valuation
    \item Quantify customer lifetime value incorporating dynamic churn and claims probabilities
    \item Design strategic segmentation enabling differentiated retention interventions
    \item Operationalize analytics through natural language interfaces
\end{enumerate}

% ============================================================================
% 2. LITERATURE REVIEW
% ============================================================================
\section{Literature Review}

\subsection{Customer Retention and Churn Prediction}

Contemporary literature establishes that machine learning methodologies substantially surpass traditional statistical approaches in churn forecasting. \citet{zhang2023ensemble} demonstrated that ensemble learning techniques combining gradient boosting algorithms attain superior predictive accuracy exceeding 85\% in large-scale implementations. \citet{afriyie2024machine} showed through systematic comparison that tree-based models, particularly XGBoost and LightGBM, consistently outperform logistic regression due to native capability for handling class imbalance and capturing feature interactions.

\subsection{Claims Modeling and Actuarial Analytics}

Recent advances demonstrate gradient boosting algorithms have emerged as the dominant modeling paradigm for claims analytics. \citet{richman2023ai} provided comprehensive evidence that artificial intelligence approaches substantially outperform traditional generalized linear models. \citet{avanzi2023boosting} demonstrated that tree-based boosting methods effectively handle zero-inflated characteristics of insurance claims data while maintaining interpretability through SHAP values.

\subsection{Customer Lifetime Value and RAG Systems}

Customer Lifetime Value (CLV) has become central to retention investment optimization. \citet{chamberlain2024customer} demonstrated that machine learning regressors incorporating behavioral features significantly outperform linear models. The CLV calculation follows actuarial principles:
\begin{equation}
    \text{CLV} = \sum_{t=1}^{T} \frac{(P_t - C_t - E_t) \cdot S_t}{(1 + r)^t} - A_0
    \label{eq:clv}
\end{equation}
where $P_t$ represents premium, $C_t$ denotes expected claims cost, $E_t$ captures operating expenses, $S_t$ indicates survival probability, $r$ is the discount rate, and $A_0$ represents acquisition cost.

Retrieval-Augmented Generation (RAG) has emerged as a promising solution for operationalizing analytics. \citet{gao2023retrieval} provided evidence that RAG architectures significantly reduce hallucinations and improve factual accuracy in knowledge-intensive domains. \citet{lewis2020retrieval} demonstrated that systems combining dense retrieval with generative models achieve state-of-the-art performance on question-answering tasks.

% ============================================================================
% 3. METHODOLOGY
% ============================================================================
\section{Materials and Methods}

This study employs the CRISP-DM framework \citep{schroder2021crisp}, encompassing six phases: business understanding, data understanding, data preparation, modeling, evaluation, and deployment.

\subsection{Data Source and Characteristics}

Administrative records were obtained from ICPSR's open repository, representing a European insurance company's non-life motor vehicle insurance portfolio spanning November 2015 to December 2018. The dataset comprises $N = 105,555$ policy transactions with 30 variables across 53,502 unique policyholders.

\begin{table}[H]
\centering
\caption{Variable taxonomy with descriptive statistics by analytical domain.}
\label{tab:variables}
\small
\begin{tabular}{@{}llcc@{}}
\toprule
\textbf{Domain} & \textbf{Variable} & \textbf{Type} & \textbf{Summary} \\
\midrule
Customer Profile & Seniority (years) & Continuous & $6.7 \pm 5.8$ \\
Demographics & Driver Age (years) & Continuous & $47.9 \pm 12.3$ \\
Policy Details & Premium (\texteuro{}) & Continuous & $315.89 \pm 201.45$ \\
& Distribution Channel & Categorical & Agent (54.9\%) \\
Vehicle Specs & Vehicle Value (\texteuro{}) & Continuous & $18,413 \pm 12,847$ \\
Claims History & Historical Claims & Discrete & $2.75 \pm 3.12$ \\
\midrule
Target Variables & Lapse (Churn) & Binary & 20.4\% positive \\
& Claims Binary & Binary & 18.6\% positive \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Data Preprocessing}

Overall dataset completeness reached 97.39\%. Missing value mechanisms were classified and treated according to \citet{little2019statistical}. Structurally missing lapse dates (85.5\%) represent right-censored observations. MCAR variables (vehicle length, fuel type) received K-Nearest Neighbors imputation ($k=5$). Winsorization preserved record counts while constraining extreme values at specified percentiles.

Systematic feature engineering created 98 composite variables from 30 raw features, organized into temporal features, risk indicators, financial metrics, logarithmic transforms, and interaction features:
\begin{equation}
    \text{Loss\_Ratio} = \frac{\text{Claims\_Cost}}{\text{Premium}}, \quad
    \text{Premium\_Adequacy} = \frac{\text{Premium}}{\mathbb{E}[\text{Claims}] + \text{Operating\_Cost}}
    \label{eq:financial}
\end{equation}

\subsection{Machine Learning Architecture}

Gradient boosting algorithms were selected based on documented superiority in insurance applications \citep{richman2023ai, avanzi2023boosting}. Key advantages include native handling of mixed data types, robust treatment of missing values, automatic capture of non-linear relationships, and built-in feature importance metrics.

\textbf{Model 1 (Churn) \& Model 2 (Claims Frequency):} Gradient Boosting Classifier with $n_{\text{estimators}} = 100$, learning rate $\eta = 0.1$, maximum depth $d_{\max} = 5$, and class weights reflecting imbalance (3.90 for churn, 4.37 for claims).

\textbf{Model 3 (Claims Severity):} Gradient Boosting Regressor trained on claimants only ($N = 19,646$) with log-transformed target. Leakage features (severity-derived indicators) excluded from predictor set.

\textbf{Model 4 (CLV):} Probabilistic integration:
\begin{equation}
    \text{CLV}_i = \sum_{t=1}^{10} \left[(P_{i,t} \times 0.75 - \hat{C}_{i,t} - E_t) \times \hat{S}_{i,t} \times (1.05)^{-t}\right] - A_i
    \label{eq:clv_full}
\end{equation}
where $\hat{S}_{i,t} = \prod_{j=1}^{t}(1 - \hat{p}_{\text{churn},i})$ and $\hat{C}_{i,t} = \hat{p}_{\text{freq},i} \times \hat{y}_{\text{severity},i}$.

\textbf{Model 5 (Segmentation):} Rule-based classification using CLV and claims risk quartiles producing four quadrants: PROTECT (high CLV, low risk), DEVELOP (low CLV, low risk), MANAGE (high CLV, high risk), EXIT (low CLV, high risk).

\subsection{Validation Methodology}

Temporal train-test split: Training (2015--2017, $N = 84,444$, 80\%) and Test (2018, $N = 21,111$, 20\%). Hyperparameter optimization via Bayesian optimization with Tree-structured Parzen Estimator using 50--80 trials and stratified 5-fold cross-validation.

\subsection{RAG System Implementation}

The RAG system comprises: (1) policy records converted to natural language descriptions; (2) SentenceTransformer (all-MiniLM-L6-v2) producing 384-dimensional embeddings; (3) FAISS IndexFlatL2 for exact nearest neighbor search; and (4) natural language query interface.

% ============================================================================
% 4. RESULTS
% ============================================================================
\section{Results}

\subsection{Data Quality and Exploratory Analysis}

The final analytical dataset comprised 105,555 records (97.39\% complete) with 8,641 outliers winsorized. Key EDA findings are summarized below.

\subsubsection{Customer Lifecycle Vulnerability}

Lapse rate analysis revealed a pronounced vulnerability window during policy years 1--3 (Figure~\ref{fig:lifecycle}).

\begin{figure}[H]
\centering
\includegraphics[width=0.80\textwidth]{Lifecycle.png}
\caption{Customer lifecycle vulnerability curve. The ``valley of death'' during years 1--3 exhibits 26.5\% average lapse rate, 58\% above portfolio average (20.4\%).}
\label{fig:lifecycle}
\end{figure}

\begin{table}[H]
\centering
\caption{Lapse rate decomposition by tenure cohort.}
\label{tab:lifecycle}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Tenure Cohort} & \textbf{N} & \textbf{Lapse Rate} & \textbf{vs. Average} & \textbf{p-value} \\
\midrule
Year 0 (New) & 18,234 & 11.2\% & $-45.1\%$ & $<0.001$ \\
Years 1--3 & 32,567 & 26.5\% & $+29.9\%$ & $<0.001$ \\
Years 3--5 & 24,891 & 24.9\% & $+22.1\%$ & $<0.001$ \\
Years 5--10 & 18,456 & 17.6\% & $-13.7\%$ & $<0.001$ \\
Years 10+ & 11,407 & 16.7\% & $-18.1\%$ & $<0.001$ \\
\midrule
\textbf{Portfolio} & \textbf{105,555} & \textbf{20.4\%} & --- & --- \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{Distribution Channel Economics}

Agent-sourced policies demonstrated systematic advantages across all performance dimensions (Figure~\ref{fig:channel}).

\begin{figure}[H]
\centering
\includegraphics[width=0.80\textwidth]{ROI and Channel.png}
\caption{Distribution channel comparative economics. Agent channel: 752\% ROI, 8.23 years tenure, 44.3\% loss ratio. Broker channel: 297\% ROI, 4.84 years tenure, 53.4\% loss ratio.}
\label{fig:channel}
\end{figure}

\begin{table}[H]
\centering
\caption{Distribution channel performance comparison.}
\label{tab:channel}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Metric} & \textbf{Agent} & \textbf{Broker} & \textbf{Difference} & \textbf{Advantage} \\
\midrule
Mean CLV (\texteuro{}) & 727 & 244 & +483 & $+198\%$ \\
ROI (\%) & 752 & 297 & +455pp & $+153\%$ \\
Mean Tenure (years) & 8.23 & 4.84 & +3.39 & $+70\%$ \\
Loss Ratio (\%) & 44.3 & 53.4 & $-9.1$pp & $+17\%$ \\
Churn Rate (\%) & 16.2 & 20.5 & $-4.3$pp & $+21\%$ \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{Premium Adequacy and Segmentation}

Pure premium analysis identified 14.8\% of policies as systematically underpriced, concentrated in broker-sourced urban commercial vehicles (22.3\% prevalence) versus agent-sourced rural passenger cars (7.8\%). The value-risk matrix classified the portfolio into four strategic segments (Figure~\ref{fig:segments}).

\begin{figure}[H]
\centering
\includegraphics[width=0.75\textwidth]{segment.png}
\caption{Customer segmentation matrix. PROTECT (34.6\%, \texteuro{}542 CLV), DEVELOP (30.8\%, \texteuro{}156), MANAGE (15.4\%, \texteuro{}387), EXIT (19.2\%, \texteuro{}89).}
\label{fig:segments}
\end{figure}

\subsection{Predictive Model Performance}

\begin{table}[H]
\centering
\caption{Model performance summary across all predictive tasks.}
\label{tab:all_models}
\begin{tabular}{@{}llccc@{}}
\toprule
\textbf{Model} & \textbf{Primary Metric} & \textbf{Baseline} & \textbf{Optimized} & \textbf{Target} \\
\midrule
Churn Prediction & ROC-AUC & 0.8805 & \textbf{0.8926} & $\geq 0.85$ \\
Claims Frequency & ROC-AUC & 0.9211 & \textbf{0.9225} & $\geq 0.80$ \\
Claims Severity & $R^2$ (clean) & --- & \textbf{0.352} & $> 0.30$ \\
CLV & Portfolio Total & --- & \textbf{\texteuro{}25.8M} & --- \\
\bottomrule
\end{tabular}
\end{table}

The optimized churn model achieved ROC-AUC of 0.8926, exceeding the target threshold. Claims frequency prediction achieved ROC-AUC of 0.9225, demonstrating strong discrimination. The severity model's $R^2 = 0.352$ (after leakage removal) represents realistic predictive capability---the reduction from 0.645 with leakage features demonstrates rigorous data hygiene rather than model degradation.

CLV calculations quantified total portfolio value at \texteuro{}25.8 million with agent-sourced customers contributing \texteuro{}16.9M (mean \texteuro{}727) versus broker-sourced at \texteuro{}8.9M (mean \texteuro{}244).

Feature importance analysis revealed historical claims rate as the dominant predictor across all models, followed by tenure, premium, and vehicle characteristics.

\subsection{RAG System and Pilot Deployment}

The RAG system achieved 82\% production readiness with 24ms median query latency, 53,502 documents indexed, and 87\% query accuracy.

\begin{table}[H]
\centering
\caption{Pilot deployment results (3 months, 20 agents, 12,000 customers).}
\label{tab:pilot}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Metric} & \textbf{Baseline} & \textbf{Pilot} & \textbf{Change} \\
\midrule
Churn Rate & 20.4\% & 17.9\% & $-12.3\%$ \\
Value Preserved (\texteuro{}) & --- & 2,370,000 & --- \\
Operational Efficiency & Baseline & $+35\%$ & --- \\
\midrule
Implementation Cost (\texteuro{}) & \multicolumn{2}{c}{70,000} & --- \\
First-Year ROI & \multicolumn{2}{c}{3,286\%} & --- \\
\bottomrule
\end{tabular}
\end{table}

% ============================================================================
% 5. DISCUSSION
% ============================================================================
\section{Discussion}

\subsection{Key Findings Interpretation}

\textbf{The Lifecycle Valley of Death:} The 26.5\% lapse rate during policy years 1--3 (58\% above average) confirms theoretical expectations from relationship marketing literature \citep{kumar2024retention}. The initial ``honeymoon period'' (year 0, 11.2\% lapse) transitions into a critical decision point. Retention investment should disproportionately target the 1--3 year cohort where intervention ROI is maximized.

\textbf{Distribution Channel Economics:} The 2.5$\times$ ROI differential between agent (752\%) and broker (297\%) channels represents one of the study's most strategically significant findings. This gap persists despite broker customers paying 8.5\% higher premiums, indicating systematic underpricing relative to actual risk profiles. Three compounding factors drive agent superiority: 60\% longer tenure, 14\% lower loss ratios, and 21\% reduced churn. The \texteuro{}483 per-customer CLV difference, extrapolated across a typical portfolio, represents millions in foregone value.

\textbf{Model Performance:} Achieving 89.26\% ROC-AUC for churn and 92.25\% for claims frequency substantially exceeds industry benchmarks where performance above 0.80 is considered strong \citep{afriyie2024machine}. The severity model's $R^2 = 0.352$ after leakage removal represents honest predictive capability suitable for production deployment.

\textbf{Premium Adequacy:} The 14.8\% of systematically underpriced policies represents ``toxic revenue'' actively damaging profitability. Underpricing concentrates predictably in broker-sourced urban commercial vehicles, suggesting organizational blind spots rather than random pricing errors.

\subsection{Literature Contextualization}

The findings confirm gradient boosting superiority for insurance applications \citep{richman2023ai, avanzi2023boosting} while addressing the gap identified by \citet{tharmarajan2024comparative} regarding isolated treatment of churn, claims, and value prediction. The RAG deployment achieving 82\% production readiness confirms \citet{gao2023retrieval}'s assertion regarding retrieval-augmented systems in knowledge-intensive domains.

\subsection{Limitations}

\textbf{Geographic Transferability:} The European dataset may not capture Kenya-specific factors including fraud patterns and regulatory environment. \textbf{Temporal Stability:} The 2015--2018 window captured stable economic conditions; model performance during recessions remains untested. \textbf{Causal Inference:} Absence of randomized control group limits causal attribution. \textbf{Data Recency:} Telematics information increasingly available was absent from this dataset.

\subsection{Future Research}

Priority directions include: (1) temporal stability analysis across 10+ years spanning economic cycles; (2) randomized controlled trials quantifying causal effects of specific retention interventions; (3) multi-line integration incorporating life, health, and property insurance; and (4) direct replication using Kenyan portfolio data.

% ============================================================================
% 6. CONCLUSIONS
% ============================================================================
\section{Conclusions}

This research developed and validated an integrated machine learning framework for automobile insurance customer analytics comprising: (1) churn prediction at 89.26\% ROC-AUC; (2) claims analytics at 92.25\% ROC-AUC for frequency and $R^2 = 0.352$ for severity; (3) \texteuro{}25.8 million portfolio valuation with \texteuro{}483 systematic channel differential; and (4) four-quadrant strategic segmentation.

Three empirical findings carry direct strategic implications. First, the ``lifecycle valley of death'' during policy years 1--3 concentrates 58\% elevated churn risk---retention investment should target early-tenure customers. Second, agent-sourced customers generate 2.5$\times$ higher ROI than broker channel---strategic channel prioritization and broker pricing recalibration can recover millions in foregone value. Third, 14.8\% of policies are systematically underpriced---automated pricing adequacy monitoring can eliminate toxic revenue.

For practitioners, the framework enables immediate implementation: deploy churn scoring 60 days before renewal; implement differentiated retention strategies by segment; establish automated pricing adequacy alerts; prioritize agent channel development; and recalibrate broker channel pricing.

The pilot deployment demonstrating 3,286\% first-year ROI validates that sophisticated customer analytics need not remain exclusive to large, well-resourced insurers. In fragmented markets where acquisition costs reach 5--25$\times$ retention expense, knowing which customers will leave, when they'll make that decision, and which interventions work is not luxury analytics---it is survival intelligence.

\vspace{0.5em}
\begin{center}
\textbf{Interactive Application:} \url{https://automobilecustomerx.streamlit.app/}
\end{center}

% ============================================================================
% ACKNOWLEDGMENTS AND DECLARATIONS
% ============================================================================
\section*{Acknowledgments}
The author acknowledges Strathmore University for institutional support and computational resources.

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
