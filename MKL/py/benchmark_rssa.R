# Rssa Benchmark Script
# Run with: Rscript benchmark_rssa.R

# Install Rssa if needed
if (!require("Rssa")) {
    install.packages("Rssa", repos="https://cloud.r-project.org")
    library(Rssa)
}

L <- 100
k <- 20
output_dir <- "benchmark_data"


# === sine_noise ===
cat("Processing sine_noise...\n")
x <- scan(paste0(output_dir, "/sine_noise_signal.csv"), quiet=TRUE)

# Run SSA
s <- ssa(x, L=L, neig=k)

# Variance explained (eigenvalues / total)
eigenvalues <- s$sigma[1:k]^2
total_var <- sum(s$sigma^2)
var_explained <- eigenvalues / total_var
write.table(var_explained, paste0(output_dir, "/sine_noise_variance_rssa.csv"), 
            row.names=FALSE, col.names=FALSE)

# Trend reconstruction (component 1 in R = component 0 in Python)
trend <- reconstruct(s, groups=list(1))$F1
write.table(trend, paste0(output_dir, "/sine_noise_trend_rssa.csv"),
            row.names=FALSE, col.names=FALSE)

# Periodic (components 2,3)
if (k >= 3) {
    periodic <- reconstruct(s, groups=list(c(2,3)))$F1
    write.table(periodic, paste0(output_dir, "/sine_noise_periodic_rssa.csv"),
                row.names=FALSE, col.names=FALSE)
}

# Full reconstruction
full_recon <- reconstruct(s, groups=list(1:k))$F1
write.table(full_recon, paste0(output_dir, "/sine_noise_full_recon_rssa.csv"),
            row.names=FALSE, col.names=FALSE)

# W-correlation matrix
wcorr_mat <- wcor(s, groups=1:k)
write.table(wcorr_mat, paste0(output_dir, "/sine_noise_wcorr_rssa.csv"),
            row.names=FALSE, col.names=FALSE, sep=",")


# === trend_seasonal ===
cat("Processing trend_seasonal...\n")
x <- scan(paste0(output_dir, "/trend_seasonal_signal.csv"), quiet=TRUE)

# Run SSA
s <- ssa(x, L=L, neig=k)

# Variance explained (eigenvalues / total)
eigenvalues <- s$sigma[1:k]^2
total_var <- sum(s$sigma^2)
var_explained <- eigenvalues / total_var
write.table(var_explained, paste0(output_dir, "/trend_seasonal_variance_rssa.csv"), 
            row.names=FALSE, col.names=FALSE)

# Trend reconstruction (component 1 in R = component 0 in Python)
trend <- reconstruct(s, groups=list(1))$F1
write.table(trend, paste0(output_dir, "/trend_seasonal_trend_rssa.csv"),
            row.names=FALSE, col.names=FALSE)

# Periodic (components 2,3)
if (k >= 3) {
    periodic <- reconstruct(s, groups=list(c(2,3)))$F1
    write.table(periodic, paste0(output_dir, "/trend_seasonal_periodic_rssa.csv"),
                row.names=FALSE, col.names=FALSE)
}

# Full reconstruction
full_recon <- reconstruct(s, groups=list(1:k))$F1
write.table(full_recon, paste0(output_dir, "/trend_seasonal_full_recon_rssa.csv"),
            row.names=FALSE, col.names=FALSE)

# W-correlation matrix
wcorr_mat <- wcor(s, groups=1:k)
write.table(wcorr_mat, paste0(output_dir, "/trend_seasonal_wcorr_rssa.csv"),
            row.names=FALSE, col.names=FALSE, sep=",")


# === multi_periodic ===
cat("Processing multi_periodic...\n")
x <- scan(paste0(output_dir, "/multi_periodic_signal.csv"), quiet=TRUE)

# Run SSA
s <- ssa(x, L=L, neig=k)

# Variance explained (eigenvalues / total)
eigenvalues <- s$sigma[1:k]^2
total_var <- sum(s$sigma^2)
var_explained <- eigenvalues / total_var
write.table(var_explained, paste0(output_dir, "/multi_periodic_variance_rssa.csv"), 
            row.names=FALSE, col.names=FALSE)

# Trend reconstruction (component 1 in R = component 0 in Python)
trend <- reconstruct(s, groups=list(1))$F1
write.table(trend, paste0(output_dir, "/multi_periodic_trend_rssa.csv"),
            row.names=FALSE, col.names=FALSE)

# Periodic (components 2,3)
if (k >= 3) {
    periodic <- reconstruct(s, groups=list(c(2,3)))$F1
    write.table(periodic, paste0(output_dir, "/multi_periodic_periodic_rssa.csv"),
                row.names=FALSE, col.names=FALSE)
}

# Full reconstruction
full_recon <- reconstruct(s, groups=list(1:k))$F1
write.table(full_recon, paste0(output_dir, "/multi_periodic_full_recon_rssa.csv"),
            row.names=FALSE, col.names=FALSE)

# W-correlation matrix
wcorr_mat <- wcor(s, groups=1:k)
write.table(wcorr_mat, paste0(output_dir, "/multi_periodic_wcorr_rssa.csv"),
            row.names=FALSE, col.names=FALSE, sep=",")


# === nonlinear ===
cat("Processing nonlinear...\n")
x <- scan(paste0(output_dir, "/nonlinear_signal.csv"), quiet=TRUE)

# Run SSA
s <- ssa(x, L=L, neig=k)

# Variance explained (eigenvalues / total)
eigenvalues <- s$sigma[1:k]^2
total_var <- sum(s$sigma^2)
var_explained <- eigenvalues / total_var
write.table(var_explained, paste0(output_dir, "/nonlinear_variance_rssa.csv"), 
            row.names=FALSE, col.names=FALSE)

# Trend reconstruction (component 1 in R = component 0 in Python)
trend <- reconstruct(s, groups=list(1))$F1
write.table(trend, paste0(output_dir, "/nonlinear_trend_rssa.csv"),
            row.names=FALSE, col.names=FALSE)

# Periodic (components 2,3)
if (k >= 3) {
    periodic <- reconstruct(s, groups=list(c(2,3)))$F1
    write.table(periodic, paste0(output_dir, "/nonlinear_periodic_rssa.csv"),
                row.names=FALSE, col.names=FALSE)
}

# Full reconstruction
full_recon <- reconstruct(s, groups=list(1:k))$F1
write.table(full_recon, paste0(output_dir, "/nonlinear_full_recon_rssa.csv"),
            row.names=FALSE, col.names=FALSE)

# W-correlation matrix
wcorr_mat <- wcor(s, groups=1:k)
write.table(wcorr_mat, paste0(output_dir, "/nonlinear_wcorr_rssa.csv"),
            row.names=FALSE, col.names=FALSE, sep=",")


# === stock_sim ===
cat("Processing stock_sim...\n")
x <- scan(paste0(output_dir, "/stock_sim_signal.csv"), quiet=TRUE)

# Run SSA
s <- ssa(x, L=L, neig=k)

# Variance explained (eigenvalues / total)
eigenvalues <- s$sigma[1:k]^2
total_var <- sum(s$sigma^2)
var_explained <- eigenvalues / total_var
write.table(var_explained, paste0(output_dir, "/stock_sim_variance_rssa.csv"), 
            row.names=FALSE, col.names=FALSE)

# Trend reconstruction (component 1 in R = component 0 in Python)
trend <- reconstruct(s, groups=list(1))$F1
write.table(trend, paste0(output_dir, "/stock_sim_trend_rssa.csv"),
            row.names=FALSE, col.names=FALSE)

# Periodic (components 2,3)
if (k >= 3) {
    periodic <- reconstruct(s, groups=list(c(2,3)))$F1
    write.table(periodic, paste0(output_dir, "/stock_sim_periodic_rssa.csv"),
                row.names=FALSE, col.names=FALSE)
}

# Full reconstruction
full_recon <- reconstruct(s, groups=list(1:k))$F1
write.table(full_recon, paste0(output_dir, "/stock_sim_full_recon_rssa.csv"),
            row.names=FALSE, col.names=FALSE)

# W-correlation matrix
wcorr_mat <- wcor(s, groups=1:k)
write.table(wcorr_mat, paste0(output_dir, "/stock_sim_wcorr_rssa.csv"),
            row.names=FALSE, col.names=FALSE, sep=",")


# === Forecasting (trend_seasonal) ===
cat("Forecasting trend_seasonal...\n")
x <- scan(paste0(output_dir, "/trend_seasonal_signal.csv"), quiet=TRUE)
s <- ssa(x, L=L, neig=k)

# Forecast using components 1,2,3 (R indexing)
forecast_result <- rforecast(s, groups=list(c(1,2,3)), len=50)
write.table(forecast_result, paste0(output_dir, "/trend_seasonal_forecast_rssa.csv"),
            row.names=FALSE, col.names=FALSE)

cat("\nRssa results saved to", output_dir, "\n")
