##################################################
# Code by Mubanga Nsofu, 18.09.2025
# German_Credit script with enhanced ROC and Precision-Recall curve plotting 
# using Okabe–Ito color palette.
#
##################################################
# Load libraries ----------------------------------------------------------------

library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(mlr3mbo)          # Bayesian optimiser
library(mlr3pipelines)    # one-hot encoding
library(mlr3viz)
library(ggplot2)
library(rgenoud)
library(pROC)


# 1) Task + split --------------------------------------------------------------

task <- tsk("german_credit")   # target levels typically: "good","bad"
set.seed(123)
train_ids <- sample(task$nrow, 0.7 * task$nrow)
test_ids  <- setdiff(seq_len(task$nrow), train_ids)

# 2) Pipeline: one-hot encode -> XGBoost --------------------------------------

pipe <- po("encode", method = "one-hot") %>>%
  lrn("classif.xgboost", predict_type = "prob")
learner <- GraphLearner$new(pipe)

# 3) Search space (old paradox style): name with full inner-learner IDs --------

space <- paradox::ps(
  "classif.xgboost.nrounds"          = paradox::p_int(100, 600),
  "classif.xgboost.eta"              = paradox::p_dbl(0.01, 0.30),
  "classif.xgboost.max_depth"        = paradox::p_int(3, 10),
  "classif.xgboost.subsample"        = paradox::p_dbl(0.50, 1.00),
  "classif.xgboost.colsample_bytree" = paradox::p_dbl(0.50, 1.00),
  "classif.xgboost.gamma"            = paradox::p_dbl(0.00, 5.00),
  "classif.xgboost.lambda"           = paradox::p_dbl(0.00, 5.00),
  "classif.xgboost.alpha"            = paradox::p_dbl(0.00, 5.00),
  "classif.xgboost.min_child_weight" = paradox::p_dbl(1.00, 10.00)
)

# 4) Tuning setup --------------------------------------------------------------

resampling <- rsmp("cv", folds = 5)
measure    <- msr("classif.auc")       # threshold-free, version-safe
terminator <- trm("evals", n_evals = 40)
tuner      <- tnr("mbo")               # Bayesian optimisation

at <- AutoTuner$new(
  learner      = learner,
  resampling   = resampling,
  measure      = measure,
  search_space = space,
  terminator   = terminator,
  tuner        = tuner
)

# 5) Train tuned model + predict on holdout ------------------------------------

at$train(task, row_ids = train_ids)
pred <- at$predict(task, row_ids = test_ids)

# 6) AUC -----------------------------------------------------------------------

auc <- pred$score(msr("classif.auc"))
cat(sprintf("\nAUC (test) = %.3f\n", auc))

# Robust default-threshold metrics ----------------
# Use logical sums (no fragile table indexing). Positive class = "bad".

lvl   <- c("good","bad")
truth <- factor(pred$truth, levels = lvl)
p_bad <- pred$prob[, "bad"]

# Default threshold 0.50
yhat_05 <- factor(ifelse(p_bad >= 0.50, "bad", "good"), levels = lvl)

TP <- sum(truth == "bad"  & yhat_05 == "bad")
FP <- sum(truth == "good" & yhat_05 == "bad")
FN <- sum(truth == "bad"  & yhat_05 == "good")
TN <- sum(truth == "good" & yhat_05 == "good")

cm_05 <- matrix(c(TP, FN, FP, TN), nrow = 2, byrow = TRUE,
                dimnames = list(truth = c("bad","good"),
                                predicted = c("bad","good")))
cat("\nConfusion matrix at th=0.50:\n"); print(cm_05)

recall_05    <- ifelse((TP + FN) > 0, TP / (TP + FN), NA_real_)
precision_05 <- ifelse((TP + FP) > 0, TP / (TP + FP), NA_real_)
f1_05        <- ifelse((precision_05 + recall_05) > 0,
                       2 * precision_05 * recall_05 / (precision_05 + recall_05),
                       NA_real_)
cat(sprintf("Default th=0.50 -> Recall(bad)=%.3f | Precision(bad)=%.3f | F1(bad)=%.3f\n",
            recall_05, precision_05, f1_05))

# Recall-first threshold sweep --------------------
# Aim for high recall (e.g., >= 0.90), then pick the threshold with best F1.

target_recall <- 0.90
ths <- seq(0.05, 0.95, by = 0.01)

metrics_at <- function(th) {
  yhat <- factor(ifelse(p_bad >= th, "bad", "good"), levels = lvl)
  
  TP <- sum(truth == "bad"  & yhat == "bad")
  FP <- sum(truth == "good" & yhat == "bad")
  FN <- sum(truth == "bad"  & yhat == "good")
  TN <- sum(truth == "good" & yhat == "good")
  
  rec <- ifelse((TP + FN) > 0, TP / (TP + FN), NA_real_)
  prec<- ifelse((TP + FP) > 0, TP / (TP + FP), NA_real_)
  f1  <- ifelse((prec + rec) > 0, 2 * prec * rec / (prec + rec), NA_real_)
  
  cm <- matrix(c(TP, FN, FP, TN), nrow = 2, byrow = TRUE,
               dimnames = list(truth = c("bad","good"),
                               predicted = c("bad","good")))
  list(th = th, precision = prec, recall = rec, f1 = f1, cm = cm)
}

res <- lapply(ths, metrics_at)
ok  <- vapply(res, function(x) !is.na(x$recall) && x$recall >= target_recall, logical(1))

cat("\n--- Recall-first threshold search ---\n")
if (!any(ok)) {
  cat(sprintf("No threshold achieved recall >= %.2f\n", target_recall))
} else {
  cand <- res[ok]
  f1s  <- vapply(cand, function(x) x$f1, numeric(1))
  best <- cand[[ which.max(f1s) ]]
  
  cat(sprintf("Chosen threshold: %.2f\n", best$th))
  cat(sprintf("Recall: %.3f | Precision: %.3f | F1: %.3f\n",
              best$recall, best$precision, best$f1))
  cat("\nConfusion matrix at chosen threshold:\n"); print(best$cm)
  
  # Compare to default threshold
  base <- metrics_at(0.50)
  cat("\n--- Comparison (0.50 vs chosen) ---\n")
  cat(sprintf("0.50 -> Recall: %.3f | Precision: %.3f | F1: %.3f\n",
              base$recall, base$precision, base$f1))
  cat(sprintf("%.2f -> Recall: %.3f | Precision: %.3f | F1: %.3f\n",
              best$th, best$recall, best$precision, best$f1))
}

# 10) ROC plot -----------------------------------------------------------------


# Compute ROC
roc_obj <- pROC::roc(response = truth, predictor = p_bad,
                     levels = c("good", "bad"), direction = "<")

roc_df <- data.frame(
  fpr = 1 - roc_obj$specificities,
  tpr = roc_obj$sensitivities
)

auc_num <- as.numeric(pROC::auc(roc_obj))

# Okabe–Ito palette 
okabe_ito <- c(
  blue = "#0072B2",      # main curve
  vermillion = "#D55E00",# baseline
  green = "#009E73",     # τ=0.50 point
  purple = "#CC79A7",    # τ=0.15 point
  gray = "#999999"
)

roc_df |> 
ggplot(aes(fpr, tpr)) +
  geom_line(color = okabe_ito[1], linewidth = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed",
              color = okabe_ito[2], linewidth = 0.8) +
  coord_equal() +
  labs(
    title = sprintf("ROC Curve (AUC = %.3f)", auc_num),
    x = "False Positive Rate",
    y = "True Positive Rate",
    caption = "Dashed line: Random classifier, solid line: XGBoost model\nplotted by M.Nsofu, 18.9.2025"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    panel.grid.major = element_line(color = "grey85", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    axis.title = element_text(face = "bold")
  )


# ---- Precision–Recall (PR) curve with Okabe–Ito, no warnings ----


# 11) Compute PR points over thresholds (unique scores + 0/1 guards)

ths <- sort(unique(c(0, p_bad, 1)), decreasing = TRUE)

pr_points <- lapply(ths, function(th) {
  yhat_bad <- (p_bad >= th)
  TP <- sum(truth == "bad"  & yhat_bad)
  FP <- sum(truth == "good" & yhat_bad)
  FN <- sum(truth == "bad"  & !yhat_bad)
  
  precision <- ifelse((TP + FP) > 0, TP / (TP + FP), NA_real_)
  recall    <- ifelse((TP + FN) > 0, TP / (TP + FN), NA_real_)
  c(th = th, precision = precision, recall = recall)
})

pr_df <- as.data.frame(do.call(rbind, pr_points))
pr_df <- pr_df[complete.cases(pr_df[, c("precision","recall")]), ]
pr_df <- pr_df[order(pr_df$recall), ]   # sort by recall for plotting/integration

# 12) Average Precision (AP) via trapezoidal rule on (recall, precision)

trapz <- function(x, y) sum(diff(x) * (head(y, -1) + tail(y, -1)) / 2)
ap <- trapz(pr_df$recall, pr_df$precision)

# Baseline precision = prevalence of "bad"
prevalence <- mean(truth == "bad")

# 13) Plot

pr_df |> 
ggplot( aes(x = recall, y = precision)) +
  geom_line(linewidth = 1.2, color = okabe_ito["blue"]) +
  geom_hline(yintercept = prevalence, linetype = "dashed",
             color = okabe_ito["vermillion"], linewidth = 0.8) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
  labs(
    caption = "Created by M.Nsofu, 18.09.2025 • Data Source: German Credit Data",
    title = sprintf("Precision–Recall Curve (AP = %.3f)", ap),
    subtitle = sprintf("Baseline (prevalence of 'bad') = %.3f", prevalence),
    x = "Recall (Sensitivity for 'bad')",
    y = "Precision (PPV for 'bad')"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title  = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 11, color = okabe_ito["gray"]),
    panel.grid.major = element_line(color = "grey85", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    axis.title = element_text(face = "bold")
  )
# End of PR curve code