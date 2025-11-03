#!/bin/bash

pip install -e ".[benchmark,test]"
python demo.py
python run_tests.py
bash benchmark_tools/benchmark_allcombs.sh
streamlit run benchmark_tools/streamlit_app.py

exit 0