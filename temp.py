# Model Training
# ... (training code from step 1) ...

# Model Evaluation (Optional)
carbon_percent_score = model_carbon_percent.score(X_test, y_carbon_percent_test)
chromium_percent_score = model_chromium_percent.score(X_test, y_chromium_percent_test)
# ... similar evaluation for other elements ...

# Save Trained Models
# ... (saving code) ...
