# ===== OPTIONAL LOG/CLIP TRANSFORMATION (kept from your "below" block; commented) =====
# Apply log1p to avoid issues with zero values (log1p(x) = log(1 + x))
# X_train$creatinine_phosphokinase      <- pmin(X_train$creatinine_phosphokinase, 3000)
# X_validation$creatinine_phosphokinase <- pmin(X_validation$creatinine_phosphokinase, 3000)
# X_test$creatinine_phosphokinase       <- pmin(X_test$creatinine_phosphokinase, 3000)
