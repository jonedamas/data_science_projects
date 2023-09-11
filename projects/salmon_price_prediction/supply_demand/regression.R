# Supply model regression

supply_data <- read.csv(
  "projects/salmon_price_prediction/supply_demand/supply_data.csv",
  header = TRUE
)

reg_model <- lm(
  supply_data$Price ~
    supply_data$Volume +
      supply_data$ZW.F +
      supply_data$ZS.F,
  data = supply_data
)
summary(reg_model)
