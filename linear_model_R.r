#!/usr/bin/env Rscript

required_packages <- c("readr", "ggplot2", "dplyr", "ggrepel")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

# Load required libraries
suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
  library(dplyr)
  library(ggrepel)
})

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
  cat("Usage: Rscript linear_regression_combined.R <filename> <x_column> <y_column>\n")
  quit(status = 1)
}

filename <- args[1]
x_col <- args[2]
y_col <- args[3]

# Read and clean the data
if (!file.exists(filename)) {
  cat(sprintf("Error: File '%s' not found.\n", filename))
  quit(status = 1)
}

data <- read_csv(filename, show_col_types = FALSE)
colnames(data) <- trimws(colnames(data))

if (!(x_col %in% names(data)) || !(y_col %in% names(data))) {
  cat(sprintf("Error: Column '%s' or '%s' not found in the file.\n", x_col, y_col))
  cat(sprintf("Available columns: %s\n", paste(names(data), collapse = ", ")))
  quit(status = 1)
}

# Fit the linear model
formula <- as.formula(paste(y_col, "~", x_col))
model <- lm(formula, data = data)
predictions <- predict(model, newdata = data)

# Extract regression statistics
slope <- coef(model)[[2]]
intercept <- coef(model)[[1]]
r_squared <- summary(model)$r.squared
mse <- mean((data[[y_col]] - predictions)^2)

# Create annotation text
equation <- sprintf("y = %.2fx + %.2f", slope, intercept)
r2_text <- sprintf("  R² = %.3f", r_squared)
annotation_text <- paste(equation, r2_text, sep = "\n")

# Prepare the data for plotting
plot_data <- data %>%
  mutate(Predicted = predict(model, newdata = data))

# Plot with ggplot2
p <- ggplot(plot_data, aes_string(x = x_col, y = y_col)) +
  geom_point(aes(color = "Actual data"), size = 2) +
  geom_line(aes_string(y = "Predicted", color = shQuote("Regression line")), size = 1) +
  scale_color_manual(values = c("Actual data" = "red", "Regression line" = "blue")) +
  annotate("text", x = -Inf, y = Inf, label = annotation_text,
           hjust = -0.1, vjust = 1.5, size = 5, fontface = "italic",
           color = "black", parse = FALSE) +
  labs(
    title = paste(y_col, "vs", x_col),
    x = x_col,
    y = y_col,
    color = "Legend"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = c(0.95, 0.05),
    legend.justification = c(1, 0),
    legend.background = element_rect(fill = "white", color = "black")
  )

ggsave("regression_plot_r.png", plot = p, width = 8, height = 6, dpi = 300)
print(p)

# Print regression statistics
cat("\nLinear Regression Results:\n")
cat(sprintf("Slope = %.4f\n", slope))
cat(sprintf("y-intercept = %.4f\n", intercept))
cat(sprintf("R-squared = %.3f\n", r_squared))
cat(sprintf("MSE = %.2f\n", mse))