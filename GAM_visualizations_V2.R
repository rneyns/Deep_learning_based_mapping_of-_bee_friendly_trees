library(mgcv)
library(ggplot2)

# Load Data
file_path <- 'final_model_for_gpb12.csv'
data <- read.csv(file_path)

# Convert year to a factor
data$year <- as.factor(data$year)

# Fit GAM with the same smooth for all years, but allowing random variation in year
gam_model <- gam(aggregation_size ~ s(X500, k=5) + s(year, bs = "re"), 
                 family = nb(), data = data)

# Predict on the response scale
data$gam_pred <- predict(gam_model, type = "response")

# Compute confidence intervals
preds <- predict(gam_model, type = "link", se.fit = TRUE)
data$lower_ci <- exp(preds$fit - 1.96 * preds$se.fit)
data$upper_ci <- exp(preds$fit + 1.96 * preds$se.fit)

# Plot: Facet by year but keep the same smooth function
ggplot(data, aes(x = X500, y = aggregation_size)) +
  geom_point(color = "blue", alpha = 0.3, size = 1) +            # Data points
  geom_line(aes(y = gam_pred), color = "red", size = 1) +        # Same smooth function
  geom_ribbon(aes(ymin = lower_ci, ymax = upper_ci), 
              alpha = 0.15, fill = "red") +                      # Confidence interval
  labs(title = "Relationship between Aggregation Size and Salix Crown Volume",
       x = "Volume of Salix Crown within 500m Buffer [m³]",
       y = "Aggregation Size") +
  facet_wrap(~ year) +                                           # Separate plots per year
  theme_minimal()


summary(gam_model)
