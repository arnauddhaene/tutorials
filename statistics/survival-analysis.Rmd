---
title: "A Survival Analysis of Primary Biliary Cirrhosis"
author: "Arnaud Dhaene"
date: "Applied Biostatistics Spring 2020"
output:
  pdf_document:
    fig_caption: yes
bibliography: references.bib
csl: ieee.csl
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
knitr::opts_chunk$set(fig.width=6, fig.height=3.7, fig.align="center")
```


``` {r libraries, echo=FALSE, message=FALSE}
library(purrr)
library(survival)
library(gridExtra)
library(packHV)
library(dplyr)
library(plyr)
library(survminer)
library(ggplot2)
library(coin)
library(gtsummary)
library(huxtable)
```

## Introduction

Primary biliary cholangitis (PBC), previously know as primary biliary cirrhosis, is a chronic inflammatory autoimmune disease of the liver. Characterized by clinical homogeneity and an overwhelming female predominance, the disease progresses to cirrhosis and finally to liver failure over the span of 10-20 years [@poupon_primary_2010]. The prevalence being less than 1/2000, PBC is predominant in postmenopausal females. In fact, it has an estimated female-to-male ratio of 10 to 1 [@hirschfield_immunobiology_2013]. In the span of ten years between 1974 and 1984, the Mayo Clinic conducted a double-blinded randomized trial containing survival information for 312 patients. The effect of a drug, D-penicillamine (DPCA), on PBC was compared with a placebo. Liver transplantation is considered—to date—as the only effective therapy for PBC. As liver transplant success rate has increased and is now commonly used to treat PBC, the Mayo Clinic data is one of the last containing natural survival data for the disease [@fleming_counting_2005]. In this review, I provide a survival analysis of the Mayo Clinic data, first by evaluating the effect of DPCA and thereafter by using demographic, clinical, biochemical and histologic measurements to create a survival regression model.

## Models and Methods

### Data

A total of 312 patients participated in the randomized trial. An additional 106 cases did not participate, but had basic measurements recorded. Therefore, the complete dataset contains survival information for 418 patients. The observed time variable is considered as the number of days between registration and the earlier of death, transplantation, or study analysis time in July, 1986. The randomized trial data contains the following demographic, clinical, biochemical and histologic measurements:

* drug (DPCA or placebo)
* age*
* sex*
* prescence of asictes
* presence of hepatomegaly
* presence of spiders
* presence of edema*
* serum bilirubin [mg/dl]*
* serum cholesterol [mg/dl]
* albumin [gm/dl]*
* urine copper [ug/day]
* alkaline phosphatase in [U/liter]
* serum glutamic-oxaloacetic transaminase (SGOT) [U/ml]
* triglicerides [mg/dl]
* platelets per cubic ml / 1000*
* prothrombin time [s]*
* histologic stage of disease (I - portal stage, II - periportal stage, III - septal stage, and IV - biliary cirrhoses [@ludwig_staging_1978])*

The parameters marked with a star are also present for the additional 106 patients that did not participate in the randomized trial.

```{r Load Data, echo=FALSE, message=FALSE}
# Load data from given website
PBC <- read.fwf("http://lib.stat.cmu.edu/datasets/pbc",
                widths=c(3, 5, 2, 2, 6, 2, 2, 2, 2, 5, 4, 5, 5, 4, 8, 7, 4, 4, 5, 2),
                skip=60,
                col.names=c('ID', 'DAYS', 'STATUS', 'DRUG', 'AGE', 'SEX', 'ASCITES', 'HEPATOM', 'SPIDERS', 'EDEMA', 'BILI', 'CHOL', 'ALBUMIN', 'COPPER', 'PHOS', 'SGOT', 'TRIG', 'PLATELET', 'PROTIME', 'STAGE'))

cols = c(11, 12, 13, 14, 15, 16, 17, 18, 19)
PBC[,cols] = apply(PBC[,cols], 2, function(x) suppressWarnings(as.numeric(as.character(x))))

# Make sex and edema factor
PBC$SEX <- as.factor(PBC$SEX)

PBC$SEX <- mapvalues(PBC$SEX, from = c(0, 1), to = c("Male", "Female"))
PBC$ASCITES <- mapvalues(PBC$ASCITES, from = c(" .", " 0", " 1"), to = c("NA", "Absence", "Presence"))
PBC$HEPATOM <- mapvalues(PBC$HEPATOM, from = c(" .", " 0", " 1"), to = c("NA", "Absence", "Presence"))
PBC$SPIDERS <- mapvalues(PBC$SPIDERS, from = c(" .", " 0", " 1"), to = c("NA", "Absence", "Presence"))
PBC$STAGE <- mapvalues(PBC$STAGE, from = c(" .", " 1", " 2", " 3", " 4"), to = c("NA", "I", "II", "III", "IV"))


PBC$EDEMA <- as.factor(PBC$EDEMA)
PBC$EDEMA <- mapvalues(PBC$EDEMA, from = c(0, 0.5, 1), to = c("Absence", "Resolved", "Presence"))

PBCraw <- PBC

# Status is handled differently for Surv object
PBC$STATUS <- as.factor(PBC$STATUS)
PBC$STATUS <- mapvalues(PBC$STATUS, from = c("0", "1", "2"), to = c("cens.", "tx", "death"))
PBC$DRUG <- mapvalues(PBC$DRUG, from = c(" .", " 1", " 2"), to = c("NA", "DPCA", "Placebo"))

# Age in Years
PBC$AGE <- PBC$AGE / 365.25

```

### Kaplan-Meier survival function estimation

The Kaplan-Meier estimator, often called **product limit estimator**, is calculated as follows. Survival times \(t_{(1)}, ..., t_{(n)}\) are ordered, \(r_j\) is the number of individuals at risk just before \(t_{(j)}\), and \(d_j\) is the number of individuals experiencing the event at time \(t_{(j)}\). The estimator relies on the following assumptions:

* censoring is unrelated to prognosis,
* survival probabilities are the identical for patients monitored early and late in the study, and
* events happened at the times specified [@efron_logistic_1988].

\begin{equation}
	\hat S (t) = \prod_{j:t_{(j)} \leq t} \left( 1 - \frac{d_j}{r_j} \right)
\end{equation}

### Log-rank hypothesis test of equality

The log-rank test, also called **Mantel-Cox test** is used to carry out a formal hypothesis test to compare two survival distributions. The test is nonparametric, widely used in clinical trials and relies on the same assumptions as the Kaplan-Meier estimator described above [@bland_logrank_2004]. The expected number of deaths for each unique death time in the data is computed and subsequently compared to the observed number of deaths using a Chi-squared test.

In this analysis, the Kaplan-Meier survival function is estimated for both treated and control patients. The patients having undergone liver transplantation are excluded from this estimation. The survival functions are then tested using the log-rank hypothesis test to evaluate the effectiveness of DCPA.

### Cox proportional-hazards model

The above mentioned methods—Kaplan-Meier estimator and log-rank test—are for univariate analysis. In contrast, the Cox Proportional-Hazards (Coxph) model [@cox_regression_1972] is a multivariate analysis tool that creates a regression model for survival time using both quantitative and categorical predictor variables. The model can be useful to asses the effect of potential risk factors on survival time. It can be written as follows, where \(t\) is the survival time, \(h(t)\) is the hazard function determined by \(n\) covariates \((x_1, x_2, ..., x_n)\)—each carrying a coefficient \(\beta_i \forall i \in [1,n]\), and \(h_0\) is the baseline hazard.

\begin{equation}
	h(t) = h_0 (t) \cdot \text{exp}(\beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n)
\end{equation}

In this report, a Cox proportional-hazards model is computed in order to find a predictor function for hazard parameters with regards to PBC.

### Exploratory Data Analysis

To extract the main characteristics of the dataset and get a visual representation of the different parameters measured, an exploratory data analysis is performed. The first step is summarizing the data. As can be observed below, the dataset consists of $89.5\%$ females. Out of 418 observed patients, 25 received liver transplant and 161 passed away during the study. Furthermore, $71.5\%$ of the studied patients' PBC was qualified as stage III or IV. The discrete clinical measurement are additionnally summarized and grouped by stage.

**Table 1: Summary of demographic variables.**

```{r summary-demographics, echo=FALSE }
summary(select(PBC, DAYS, AGE, SEX, DRUG, STAGE, STATUS))
```

**Table 2: Summary of discrete clinical, biochemical and histologic measurements.**

```{r summary-scientific, echo=FALSE, message=FALSE}
t <- tbl_summary(select(PBC, ASCITES, HEPATOM, SPIDERS, EDEMA, STAGE), by=STAGE) %>%
  modify_header(label = "**Stage**") %>%
  bold_labels()

t

```

Next, the distribution of age is visualized and grouped by treatment, sex, stage and status. As can be observed in Figure \ref{fig:age-histograms}C, severe stage of the disease is distributed more prominently for higher ages. It can also be observed in Figure \ref{fig:age-histograms}D that death is more prominent for older patients.

```{r age-histograms, echo=FALSE, fig.cap="Age distribution grouped by treatment (A), sex (B), stage (C), and status (D)."}

histo.DRUG <- ggplot(PBC, aes(x=AGE, fill=DRUG)) +
  geom_histogram(binwidth=2.5, color="black", alpha=0.5) +
  ggtitle("A") +
  xlab("Age (years)") + ylab("Count") +
  scale_fill_discrete(name = "Treatment", labels = c("NA", "DPCA", "Placebo")) +
  theme(legend.position="bottom", text=element_text(size=9),
        legend.title = element_text(size = 8),
        legend.text = element_text(size = 8),
        legend.key.size = unit(0.2, "cm")) +
  scale_x_continuous(breaks = seq(20, 80, by = 5))

histo.SEX <- ggplot(PBC, aes(x=AGE, fill=SEX)) +
  geom_histogram(binwidth=2.5, color="black", alpha=0.5) +
  ggtitle("B") +
  xlab("Age (years)") + ylab("Count") +
  scale_fill_discrete(name = "Sex", labels = c("Male", "Female")) +
  theme(legend.position="bottom", text=element_text(size=9),
        legend.title = element_text(size = 8),
        legend.text = element_text(size = 8),
        legend.key.size = unit(0.2, "cm")) +
  scale_x_continuous(breaks = seq(20, 80, by = 5))

histo.STAGE <- ggplot(PBC, aes(x=AGE, fill=STAGE)) +
  geom_histogram(binwidth=10, color="black", alpha=0.5, position="dodge") +
  ggtitle("C") +
  xlab("Age (years)") + ylab("Count") +
  scale_fill_discrete(name = "Stage", labels = c("NA", "I", "II", "III", "IV")) +
  theme(legend.position="bottom", text=element_text(size=9),
        legend.title = element_text(size = 8),
        legend.text = element_text(size = 8),
        legend.key.size = unit(0.2, "cm")) +
  scale_x_continuous(breaks = seq(20, 80, by = 10))

histo.STATUS <- ggplot(PBC, aes(x=AGE, fill=STATUS)) +
  geom_histogram(binwidth=5, color="black", alpha=0.5, position="dodge") +
  ggtitle("D") +
  xlab("Age (years)") + ylab("Count") +
  scale_fill_discrete(name = "Status", labels = c("Censored", "Transplant", "Death")) +   
  theme(legend.position="bottom", text=element_text(size=9),
        legend.title = element_text(size = 8),
        legend.text = element_text(size = 8),
        legend.key.size = unit(0.2, "cm")) +
  scale_x_continuous(breaks = seq(20, 80, by = 5))

grid.arrange(histo.DRUG, histo.SEX, histo.STAGE, histo.STATUS, nrow=2)

```

The distributions of biochemical and histological measurements are visualized.

```{r histological-histograms, echo=FALSE, fig.cap="Combined histograms and boxplots of the following biochemical and histologic measurements: (A) serum cholesterol, (B) albumin, (C) copper, (D) alkaline phosphatase, (E) serum glutamic-oxaloacetic transaminase, (F) triglicerides, (G) platelets, and (H) prothrombin time."}

par(mfrow=c(2,4))

histo.CHOL <- hist_boxplot(PBC$CHOL, xlab="serum cholesterol in mg/dl", main="A")
histo.ALBUMIN <- hist_boxplot(PBC$ALBUMIN, xlab="albumin in gm/dl", main="B")
histo.COPPER <- hist_boxplot(PBC$COPPER, xlab="urine copper in ug/day", main="C")
histo.PHOS <- hist_boxplot(PBC$PHOS, xlab="alkaline phosphatase in U/liter", main="D")
histo.SGOT <- hist_boxplot(PBC$SGOT, xlab="SGOT in U/ml", main="E")
histo.TRIG <- hist_boxplot(PBC$TRIG, xlab="triglicerides in mg/dl", main="F")
histo.PLATELET <- hist_boxplot(PBC$PLATELET, xlab="platelets per cubic ml / 1000", main="G")
histo.PROTIME <- hist_boxplot(PBC$PROTIME, xlab="prothrombin time in seconds", main="H")

```

The next step consists of plotting some histological measurements while comparing distributions of censored versus dead patients.

```{r status-histograms, echo=FALSE, message=FALSE, fig.cap="Prothrombin time (A), Serum bilirubin (B), Platelet (C), and Serum albumin (D) distributions grouped by survival status."}

wotx <- PBC %>% filter(STATUS != "tx")

histo.PROTIME <- ggplot(wotx, aes(x=PROTIME, fill=STATUS)) +
  geom_histogram(binwidth=0.6, color="black", alpha=0.5, position="identity") +
  ggtitle("A") +
  xlab("Prothrombin time (seconds)") + ylab("Count") +
  scale_fill_discrete(name = "Status", labels = c("Censored", "Death")) +
  theme(legend.position="bottom", text=element_text(size=9),
        legend.title = element_text(size = 8),
        legend.text = element_text(size = 8),
        legend.key.size = unit(0.2, "cm")) +
  scale_x_continuous(breaks = seq(0, 20, by = 1.2))

histo.BILI <- ggplot(wotx, aes(x=BILI, fill=STATUS)) +
  geom_histogram(binwidth=5, color="black", alpha=0.5, position="identity") +
  ggtitle("B") +
  xlab("Serum bilirubin (mg/dl)") + ylab("Count") +
  scale_fill_discrete(name = "Status", labels = c("Censored", "Death")) +
  theme(legend.position="bottom", text=element_text(size=9),
        legend.title = element_text(size = 8),
        legend.text = element_text(size = 8),
        legend.key.size = unit(0.2, "cm")) +
  scale_x_continuous(breaks = seq(0, 50, by = 5))

histo.PLATELET <- ggplot(wotx, aes(x=PLATELET, fill=STATUS)) +
  geom_histogram(binwidth=50, color="black", alpha=0.5, position="identity") +
  ggtitle("C") +
  xlab("Prothrombin time (seconds)") + ylab("Count") +
  scale_fill_discrete(name = "Status", labels = c("Censored", "Death")) +
  theme(legend.position="bottom", text=element_text(size=9),
        legend.title = element_text(size = 8),
        legend.text = element_text(size = 8),
        legend.key.size = unit(0.2, "cm")) +
  scale_x_continuous(breaks = seq(0, 700, by = 50))

histo.ALBUMIN <- ggplot(wotx, aes(x=ALBUMIN, fill=STATUS)) +
  geom_histogram(binwidth=0.25, color="black", alpha=0.5, position="identity") +
  ggtitle("D") +
  xlab("Serum bilirubin (mg/dl)") + ylab("Count") +
  scale_fill_discrete(name = "Status", labels = c("Censored", "Death")) +
  theme(legend.position="bottom", text=element_text(size=9),
        legend.title = element_text(size = 8),
        legend.text = element_text(size = 8),
        legend.key.size = unit(0.2, "cm")) +
  scale_x_continuous(breaks = seq(0, 5, by = 0.5))

suppressWarnings(grid.arrange(histo.PROTIME, histo.BILI, histo.PLATELET, histo.ALBUMIN, nrow=2))

```

## Results

### Effect of DCPA on survival

The Kaplan-Meier survival function estimation is shown in Figure \ref{fig:k-m}. It can be observed that treatment does not seem to have a significant effect on survival.

```{r Create Surv Objects, echo=FALSE}

PBCwotx <- PBCraw %>% filter(STATUS != 1)

PBCwotx$STATUS[PBCwotx$STATUS == 2] <- 1

PBCwotx$DRUG[PBCwotx$DRUG == " ."] <- " 2"

sPBCwotx <- with(PBCwotx, Surv(DAYS, STATUS))

sfPBCwotx <- survfit(data=PBCwotx, sPBCwotx ~ DRUG)

```

```{r k-m, echo=FALSE, fig.height=6, fig.cap="Survival Probability of treated (DCPA) vs. control (Placebo)."}

suppressWarnings(
  ggsurvplot(sfPBCwotx,
           xlab = "Survival Time (days)", ylab = "Survival Probability",  
           legend = "bottom", lty = c(1,2),
           legend.title = "Treatment",
           legend.labs = c("DCPA", "Placebo"),
           pval = TRUE, conf.int = FALSE,
           risk.table = TRUE, risk.table.y.text.col = T, risk.table.y.text = FALSE,
           palette = c("#ee5253", "#0abde3"),
           ggtheme = theme_light())
  )
```

The log-rank hypothesis test below validates the visual observation. This results allows us to merge the measurements from treated and non-treated patients in order to define a proportional-hazards model. Such a model can be extremely useful to evaluate the risk factors of different parameters with regards to PBC.

```{r log-rank, echo=FALSE}
survdiff(sPBCwotx ~ DRUG, data = PBCwotx)
```

### Cox proportional-hazards model

Concerning the Cox proportional-hazards model, a first model is calculated using all available parameters. The model is subsequently modified by iteratively removing the variable with the lowest Wald statistic. An intermediate model is obtained when all remaining variables have Wald statistic higher than $2.5$. The summaries of first and final steps of this procedure can be observed in Table 3.

**Table 3: Summary of first and intermediate Cox proportional-hazards models. The model using all available parameters is summarized in the 3 left-first columns and  the intermediate model with selected parameters is summarized in the 3 right-most columns.**

```{r toSurv, echo=FALSE}
toSurv <- function(x) {
  return(with(x, Surv(DAYS, STATUS)))
}

addResidualsMartingale <- function(data, coxph) {
  data$resid_mart <- residuals(coxph, type="martingale")
  return(data)
}
```

```{r cox-ph, echo=FALSE, message=FALSE}

# DF
df.all <- PBCraw %>%
  filter(STATUS != 1, BILI != " .", ALBUMIN != " .", COPPER != " .", PHOS != " .", SGOT != " .", TRIG != " .", PLATELET != " .", PROTIME != " .", EDEMA != " .", HEPATOM != " .", SPIDERS != " .", ASCITES != " .")
df.all$STATUS[df.all$STATUS == 2] <- 1
df.all$AGE = df.all$AGE / 365.25

df.int <- PBCraw %>%
  filter(STATUS != 1, BILI != " .", ALBUMIN != " .", COPPER != " .", PROTIME != " .", EDEMA != " .")  
df.int$STATUS[df.int$STATUS == 2] <- 1
df.int$AGE = df.int$AGE / 365.25

df.fin <- PBCraw %>%
  filter(STATUS != 1, BILI != " .", ALBUMIN != " .", EDEMA != " .", PROTIME != " .", COPPER != " .")
df.fin$STATUS[df.fin$STATUS == 2] <- 1
df.fin$AGE = df.fin$AGE / 365.25

# FORMULA
df.all.formula <- toSurv(df.all) ~ AGE + BILI + CHOL + ALBUMIN + COPPER + PHOS + SGOT + TRIG + PLATELET + PROTIME + EDEMA + HEPATOM + SPIDERS + ASCITES
df.int.formula <- toSurv(df.int) ~ AGE + BILI + ALBUMIN + COPPER + PROTIME + EDEMA
df.fin.formula <- toSurv(df.fin) ~ EDEMA + log(ALBUMIN) + log(BILI) + AGE:log(PROTIME) + AGE + log(COPPER)

df.all.coxph <- coxph(df.all.formula, data=df.all)
df.int.coxph <- coxph(df.int.formula, data=df.int)
df.fin.coxph <- coxph(df.fin.formula, data=df.fin)

# build survival model table
t_all <-
  df.all.coxph %>%
  tbl_regression(exponentiate = TRUE) %>%
  bold_labels()

t_int <-
  df.int.coxph %>%
  tbl_regression(exponentiate = TRUE) %>%
  bold_labels()

t_fin <-
  df.fin.coxph %>%
  tbl_regression(exponentiate = TRUE) %>%
  bold_labels()

# merge tables
tbl <-
  tbl_merge(
    tbls = list(t_all, t_int),
    tab_spanner = c("**First step**", "**Intermediate model**")
  )

tbl

```


As can be seen in Figure \ref{fig:histological-histograms}, *serum bilirubin*, *albumin*, *urine copper* and *prothrombin time* have a large amount of outliers. Moreover, the histograms show that for low values of these variables, the impact on survival is high. For these reasons, the next step consists of transforming *serum bilirubin*, *albumin*, *urine copper* and *prothrombin time* to their logarithmic values. Furthermore, the log-transformation will make these variables more normally distributed, which will increase the power of the model.

The last step consists of making $log(\text{PROTIME})$ interact with time. In fact, residual testing (c.f. Table 5 and 6) showed that there was a strong correlation between *prothrombin time* and time. This result was maintained when using the log-transformation. Adding an interaction with time is a common approach to making sure the Cox model assumptions remain valid.

The final model is summarized in Figure \ref{fig:cox-ph-forest}. It has a concordance of $0.852$ and a Wald test of $201.2$  on 7 degrees of freedom. The model indicates that the risk factors for PBC are the following:

* Presence of edema despite diuretic therapy
* Prothrombin time
* Serum bilirubin
* Urine copper

It also indicates that albumin seems to decrease risk. Another observation is that age seems to decrease the risk ($\text{HR}_{\text{age}} = 0.93$), which is not expected.

Finally, martingale residuals were tested to assess the model and visualized in Figure \ref{fig:martingale}. The martingale residuals indicate that the cox proportional-hazards assumptions are valid for this model.

```{r cox-ph-forest, echo=FALSE, fig.cap="Hazard ratio of Cox Proportional Hazards model."}
ggforest(df.fin.coxph, data=df.fin)
```

### Model assessment

**Table 4: Proportional hazards assumption test for the intermediate Cox regression model fit.**

```{r assumptions-int, echo=FALSE}
cox.zph(df.int.coxph)
```


**Table 5: Proportional hazards assumption test for the final Cox regression model fit.**

```{r assumptions-fin, echo=FALSE}
cox.zph(df.fin.coxph)
```

```{r martingale, echo=FALSE, message=FALSE, fig.cap="Martingale residuals of the final model.", fig.height=3}
df.all <- addResidualsMartingale(df.all, df.all.coxph)
df.all$STATUS <- as.factor(df.all$STATUS)
df.all$STATUS <- mapvalues(df.all$STATUS, from = c(0, 1), to = c("Censored", "Death"))
df.int <- addResidualsMartingale(df.int, df.int.coxph)
df.int$STATUS <- as.factor(df.int$STATUS)
df.int$STATUS <- mapvalues(df.int$STATUS, from = c(0, 1), to = c("Censored", "Death"))
df.fin <- addResidualsMartingale(df.fin, df.fin.coxph)
df.fin$STATUS <- as.factor(df.fin$STATUS)
df.fin$STATUS <- mapvalues(df.fin$STATUS, from = c(0, 1), to = c("Censored", "Death"))

plot_mart <- function(data) {
  return(ggplot(data = data, mapping = aes(x = AGE, y = resid_mart, color=STATUS)) +
           geom_point() +
           ylab("Martingale residuals") +
           theme(legend.position = "bottom") +
           ylim(-5,1)
         )
}

# grid.arrange(plot_mart(df.all) + labs(title = "A"),
#             plot_mart(df.int) + labs(title = "B"),
#             plot_mart(df.fin) + labs(title = "C"), nrow=1)

plot_mart(df.fin)

```

\newpage

## References
