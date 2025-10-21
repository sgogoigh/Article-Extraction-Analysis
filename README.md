# Article Extraction & Analysis

This project extracts textual content from a list of article URLs provided in an Excel file, performs text cleaning, and computes various NLP metrics such as sentiment, readability, and word-level statistics.

The goal is to produce a structured output with both the original input data and derived variables.

The project is implemented entirely in **Python** and leverages libraries like **requests, BeautifulSoup, NLTK, pandas**, for extraction, processing, and analysis.


## Features

1. **Automated Article Extraction**
   - Extracts article title and body text from a given URL.
   - Supports multiple HTML structures (`<article>`, `<div>` with paragraphs, `<section>`).
   - Preserves headings, paragraphs, bullet points, and blockquotes.
   - Removes navigation text, ads, and scripts.

2. **Text Cleaning and Formatting**
   - Deduplicates repeated lines or bullet points.
   - Normalizes line breaks and spacing.
   - Preserves headers and bullets for better readability.
   - Converts text to a consistent format for NLP analysis.

3. **Textual Analysis**
   - Computes sentiment scores:
     - **Positive Score**
     - **Negative Score**
     - **Polarity Score**
     - **Subjectivity Score**
   - Computes readability metrics:
     - Average sentence length
     - Percentage of complex words
     - Fog Index
   - Word-level metrics:
     - Word count
     - Average words per sentence
     - Complex word count
     - Syllables per word
     - Average word length
   - Counts personal pronouns (`I`, `we`, `my`, `ours`, `us`) while excluding unrelated words like “US.”

4. **Output Management**
   - Extracted text files are stored in a dedicated folder `TextStore`.
   - Outputs results in `Output.csv` , `Output.xlsx`


## Execution

Python packages required:

```
pip install -r requirements.txt
```

Undergoing analysis
```
python condensed_solution.py
```


> Made by Sunny Gogoi