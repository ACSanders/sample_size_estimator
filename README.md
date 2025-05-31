# Sample Size Estimator
A simple and interactive Streamlit app that estimates the sample size needed for A/B tests

## Tests Supported
1) Proportion tests (2-sample z-test)
2) Difference in group means (2-sample t-test)

## Launch the app on Streamlit
https://samplesizes.streamlit.app/

## Features
✅ Choose between comparing conversion rates or means
✅ Customize:
- Baseline value
- Expected lift or delta
- Alpha level (significance threshold)
- Statistical power (e.g. 80%, 90%)
- Alternative hypothesis test (two-sided, smaller, larger)
✅ Clear Results:
- Required sample size per group
- Explanations
- CSV download of inputs and results

## Main Python File
sample_size_estimator.py   # Main Streamlit app

## Running Locally
If you'd like to run it locally make sure you install the packages (see requirements file), download the sample_size_estimator.py file, and then run:

streamlit run sample_size_estimator.py

## License
This project is licensed under the MIT License.

## Author
Developed by A. C. Sanders
2025
