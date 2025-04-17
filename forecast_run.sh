# Define the base paths
CODES_PATH="Your code path here"
LOG_PATH="Your log path here (.log)"

# Run the Python scripts sequentially
python ${CODES_PATH}/local_model_forecasting.py >> $LOG_PATH 2>&1
python ${CODES_PATH}/local_model_forecasting_combining.py >> $LOG_PATH 2>&1
python ${CODES_PATH}/local_model_forecasting_AQI_calculation.py >> $LOG_PATH 2>&1

