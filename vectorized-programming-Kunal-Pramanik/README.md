[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/AUww1j7o)
# Lab 03: NumPy and Pandas

**Course:** DS605 Fundamentals of Machine Learning  
**Academic Year:** 2025-26 (Autumn)  

---

## Today’s Motivation
"Learning is not about memorizing facts, but about **understanding concepts that empower you to solve real problems**."

This lab introduces two of the most widely used Python libraries for data analysis and scientific computing: **NumPy** and **Pandas**.  
The goal is to understand how to create, manipulate, and analyze data efficiently using these tools.


## Objective
- Learn how to work with **NumPy arrays** for numerical computation.  
- Understand **array construction, slicing, iteration, and aggregation**.  
- Explore data typing and efficiency in NumPy.  
- Get familiar with **Pandas DataFrames and Series** for tabular data handling.  
- Perform basic **data cleaning, selection, and summarization**.  

---

## Prerequisites
- Python 3.x installed.
- Basic understanding of Python.
- `pip` installed in your terminal/command prompt.

---

## Setting Up the Environment
```bash

python3 -m venv venv

source venv/bin/activate      # On Linux/Mac

venv\Scripts\activate         # On Windows

pip install numpy 
pip install pandas
```


# Note:
    - Always prefer NumPy vectorized operations over Python loops for efficiency.
    - Pandas builds on NumPy, so many operations follow the same principles.