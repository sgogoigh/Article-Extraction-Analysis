import os
import re
import sys
import math
import string
import logging
from collections import Counter
from typing import List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords, opinion_lexicon
from nltk.tokenize import sent_tokenize, word_tokenize

try:
    _ = stopwords.words('english')
    _ = opinion_lexicon.words()
except Exception:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('opinion_lexicon')

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def load_wordlist(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith(';')]
    return lines


def get_positive_negative_dicts(master_dir='MasterDictionary'):
    pos = []
    neg = []
    pos_path = os.path.join(master_dir, 'positive-words.txt')
    neg_path = os.path.join(master_dir, 'negative-words.txt')
    if os.path.exists(pos_path) and os.path.exists(neg_path):
        logging.info("Loading positive/negative lists from MasterDictionary folder.")
        pos = load_wordlist(pos_path)
        neg = load_wordlist(neg_path)
    else:
        logging.info("MasterDictionary not found; falling back to NLTK opinion_lexicon.")
        pos = list(opinion_lexicon.positive())
        neg = list(opinion_lexicon.negative())
    pos_set = set(w.lower() for w in pos)
    neg_set = set(w.lower() for w in neg)
    return pos_set, neg_set


def get_stopwords_set(stopwords_dir='StopWords'):
    sw = set()
    if os.path.isdir(stopwords_dir):
        files = [os.path.join(stopwords_dir, f) for f in os.listdir(stopwords_dir) if os.path.isfile(os.path.join(stopwords_dir, f))]
        if files:
            logging.info("Loading stopwords from StopWords folder.")
            for fpath in files:
                try:
                    for w in load_wordlist(fpath):
                        sw.add(w.lower())
                except Exception:
                    continue
    if not sw:
        logging.info("Using NLTK stopwords as fallback.")
        sw = set(stopwords.words('english'))
    return sw


HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AssignmentBot/1.0; +https://example.com/bot)"
}


def extract_article_text(url: str) -> Tuple[str, str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            logging.warning(f"URL {url} returned status {r.status_code}.")
            return "", ""
        soup = BeautifulSoup(r.content, 'html.parser')

        title = ""
        h1 = soup.find('h1')
        if h1 and h1.get_text(strip=True):
            title = h1.get_text(strip=True)
        elif soup.find('meta', property='og:title'):
            title = soup.find('meta', property='og:title').get('content', '').strip()
        elif soup.title and soup.title.string:
            title = soup.title.string.strip()

        body = ""
        # 1) <article> tags
        article_tag = soup.find('article')
        if article_tag:
            paragraphs = [p.get_text(" ", strip=True) for p in article_tag.find_all('p') if p.get_text(strip=True)]
            if paragraphs:
                body = "\n".join(paragraphs)
        # 2) Common article divs: look for divs with many <p>
        if not body:
            candidates = []
            for div in soup.find_all(['div', 'section'], recursive=True):
                ps = div.find_all('p')
                if len(ps) >= 3:
                    text = " ".join(p.get_text(" ", strip=True) for p in ps if p.get_text(strip=True))
                    candidates.append((len(ps), text))
            if candidates:
                # pick the div with the most paragraphs
                candidates.sort(key=lambda x: x[0], reverse=True)
                body = "\n".join([p.strip() for p in candidates[0][1].split("\n") if p.strip()])
        # 3) fallback to all <p> in body
        if not body:
            ps = soup.find_all('p')
            if ps:
                body = "\n".join(p.get_text(" ", strip=True) for p in ps if p.get_text(strip=True))

        # Clean body: remove scripts/styles, headers or short nav text
        if body:
            # strip multiple short lines and navigation-like text
            lines = [line.strip() for line in body.splitlines() if line.strip() and len(line.strip()) > 20]
            body = "\n".join(lines)

        return title, body
    except Exception as e:
        logging.exception(f"Exception while extracting {url}: {e}")
        return "", ""

VOWELS = "aeiouy"


def count_syllables(word: str) -> int:
    w = word.lower()
    w = re.sub(r'[^a-z]', '', w)
    if not w:
        return 0
    groups = re.findall(r'[aeiouy]+', w)
    syllables = len(groups)
    if w.endswith("es") or w.endswith("ed"):
        if syllables > 1:
            syllables -= 1
    if w.endswith("e") and not w.endswith("le"):
        # silent e
        if syllables > 1:
            syllables -= 1
    if syllables == 0:
        syllables = 1
    return syllables


PERSONAL_PRONOUNS_PATTERN = re.compile(r'\b(I|we|We|WE|my|My|our|Our|ours|Ours|us|Us)\b')


def clean_and_tokenize(text: str):
    sentences = sent_tokenize(text)
    words = []
    for sent in sentences:
        for token in word_tokenize(sent):
            token = token.strip()
            if token:
                words.append(token)
    return sentences, words


def is_alpha_word(w: str) -> bool:
    return bool(re.match(r'^[A-Za-z]+$', w))


def analyze_text(text: str,
                 pos_set: set,
                 neg_set: set,
                 stopwords_set: set):
    raw_text = text or ""
    if not raw_text.strip():
        return {
            'POSITIVE SCORE': 0,
            'NEGATIVE SCORE': 0,
            'POLARITY SCORE': 0.0,
            'SUBJECTIVITY SCORE': 0.0,
            'AVG SENTENCE LENGTH': 0.0,
            'PERCENTAGE OF COMPLEX WORDS': 0.0,
            'FOG INDEX': 0.0,
            'AVG NUMBER OF WORDS PER SENTENCE': 0.0,
            'COMPLEX WORD COUNT': 0,
            'WORD COUNT': 0,
            'SYLLABLE PER WORD': 0.0,
            'PERSONAL PRONOUNS': 0,
            'AVG WORD LENGTH': 0.0
        }

    sentences = sent_tokenize(raw_text)
    total_sentences = len(sentences) if sentences else 1

    raw_tokens = []
    for s in sentences:
        raw_tokens.extend(word_tokenize(s))
    tokens_cleaned = []
    for t in raw_tokens:
        t_stripped = t.strip(string.punctuation)
        if t_stripped:
            tokens_cleaned.append(t_stripped)

    cleaned_words = [w for w in [w.lower() for w in tokens_cleaned] if w not in stopwords_set and is_alpha_word(w)]
    total_words_after_cleaning = len(cleaned_words)
    
    pos_score = sum(1 for w in cleaned_words if w in pos_set)
    neg_score = sum(1 for w in cleaned_words if w in neg_set)
    
    negative_score = neg_score
    positive_score = pos_score

    polarity_score = 0.0
    if (positive_score + negative_score) != 0:
        polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 1e-6)

    subjectivity_score = (positive_score + negative_score) / (total_words_after_cleaning + 1e-6)

    # Word count (after cleaning) per spec
    word_count = total_words_after_cleaning

    # Syllables per word & complex word count (>2 syllables)
    syllable_counts = [count_syllables(w) for w in cleaned_words]
    total_syllables = sum(syllable_counts)
    complex_word_count = sum(1 for s in syllable_counts if s > 2)

    # Average sentence length = number of words / number of sentences
    avg_sentence_length = (word_count / total_sentences) if total_sentences > 0 else 0.0

    # Percentage of Complex words
    percentage_complex_words = (complex_word_count / word_count) if word_count > 0 else 0.0

    # Fog Index
    fog_index = 0.4 * (avg_sentence_length + (percentage_complex_words * 100))  # Note: text_analysis doc appears to treat percentage as fraction; to match common formula use percentage*100. However the doc says Percentage of Complex words = complex_words / words. Fog = 0.4*(ASL + PCW). We'll use PCW as percentage (0-100) to match typical Gunning Fog.
    # But the doc's earlier formula seemed to indicate PCW as fraction — to exactly follow that doc, replace line above with: fog_index = 0.4 * (avg_sentence_length + (percentage_complex_words))
    # We'll follow the doc verbatim (PCW as fraction), so adjust:
    fog_index = 0.4 * (avg_sentence_length + (percentage_complex_words))

    # Avg number of words per sentence (same as avg_sentence_length)
    avg_words_per_sentence = avg_sentence_length

    # Syllable per word
    syllable_per_word = (total_syllables / word_count) if word_count > 0 else 0.0

    # Personal pronouns count using regex but exclude country 'US' detection -> already pattern only matches I/we/my/our/ours/us etc. We'll keep case-insensitive matching and exclude uppercase 'US' when it's standalone (two letter)
    pronoun_matches = re.findall(PERSONAL_PRONOUNS_PATTERN, raw_text)
    # the regex returns capture groups; count them, filter 'US' capital case out:
    personal_pronouns = 0
    for m in pronoun_matches:
        if m.upper() == 'US' and m.isupper():
            continue
        personal_pronouns += 1

    # Average word length: total characters in words / total number of words
    total_chars = sum(len(w) for w in cleaned_words)
    avg_word_length = (total_chars / word_count) if word_count > 0 else 0.0

    # Final dictionary
    result = {
        'POSITIVE SCORE': positive_score,
        'NEGATIVE SCORE': negative_score,
        'POLARITY SCORE': polarity_score,
        'SUBJECTIVITY SCORE': subjectivity_score,
        'AVG SENTENCE LENGTH': avg_sentence_length,
        'PERCENTAGE OF COMPLEX WORDS': percentage_complex_words,
        'FOG INDEX': fog_index,
        'AVG NUMBER OF WORDS PER SENTENCE': avg_words_per_sentence,
        'COMPLEX WORD COUNT': complex_word_count,
        'WORD COUNT': word_count,
        'SYLLABLE PER WORD': syllable_per_word,
        'PERSONAL PRONOUNS': personal_pronouns,
        'AVG WORD LENGTH': avg_word_length
    }
    return result


def find_columns(df: pd.DataFrame):
    cols = list(df.columns)
    url_col = None
    id_col = None
    for c in cols:
        if c.lower() in ('url', 'link', 'article_link', 'article url', 'article_link'):
            url_col = c
        if c.lower() in ('url_id', 'id', 'urlid', 'identifier'):
            id_col = c
   
    if url_col is None:
        for c in cols:
            if df[c].astype(str).str.startswith('http').any():
                url_col = c
                break
    if id_col is None:
        candidates = [c for c in cols if c != url_col]
        if candidates:
            id_col = candidates[0]
    return url_col, id_col


def main(input_xlsx='Input.xlsx', output_xlsx='Output.xlsx', output_csv='Output.csv'):
    if not os.path.exists(input_xlsx):
        logging.error(f"{input_xlsx} not found in CWD. Place the provided Input.xlsx in the working directory.")
        sys.exit(1)

    df_input = pd.read_excel(input_xlsx)
    if df_input.empty:
        logging.error("Input.xlsx appears empty.")
        sys.exit(1)

    url_col, id_col = find_columns(df_input)
    if url_col is None:
        logging.error("Could not detect URL column in Input.xlsx. Ensure there's a column containing URLs.")
        sys.exit(1)
    if id_col is None:
        logging.error("Could not detect URL_ID column. Ensure Input.xlsx contains an identifier column.")
        sys.exit(1)

    # Prepare dictionaries and stopwords
    pos_set, neg_set = get_positive_negative_dicts()
    stopwords_set = get_stopwords_set()

    # Prepare results list
    results = []
    # preserve all input columns first as required
    for idx, row in df_input.iterrows():
        url = str(row[url_col]).strip()
        url_id = str(row[id_col]).strip()
        logging.info(f"Processing row {idx+1}: URL_ID={url_id} URL={url}")

        title, body = ("","")
        if url.lower().startswith('http'):
            title, body = extract_article_text(url)
            if not title and not body:
                logging.warning(f"No content extracted for {url}. Possibly JS heavy; consider using Selenium.")
        else:
            logging.warning(f"URL value doesn't look like a URL: {url}")

        # Save extracted to text file named by URL_ID
        safe_name = re.sub(r'[^\w\-_.]', '_', url_id) or f'url_{idx+1}'
        txt_filename = f"{safe_name}.txt"
        with open(txt_filename, 'w', encoding='utf-8') as outf:
            # Save title and body (assignment said to extract only title and article text)
            if title:
                outf.write(title + "\n\n")
            if body:
                outf.write(body)
        logging.info(f"Saved extracted article to {txt_filename}")

        # Perform analysis on title + body or only body? text_analysis doc refers to article text.
        # We'll analyze title + body combined (title helps sentiment)
        full_text_for_analysis = (title + "\n\n" + body).strip()
        analysis = analyze_text(full_text_for_analysis, pos_set, neg_set, stopwords_set)

        # Create ordered output row: all input columns then the 13 derived variables in specified order
        out_row = {}
        # include all original input columns first
        for c in df_input.columns:
            out_row[c] = row[c]


        out_row['POSITIVE SCORE'] = analysis['POSITIVE SCORE']
        out_row['NEGATIVE SCORE'] = analysis['NEGATIVE SCORE']
        out_row['POLARITY SCORE'] = analysis['POLARITY SCORE']
        out_row['SUBJECTIVITY SCORE'] = analysis['SUBJECTIVITY SCORE']
        out_row['AVG SENTENCE LENGTH'] = analysis['AVG SENTENCE LENGTH']
        out_row['PERCENTAGE OF COMPLEX WORDS'] = analysis['PERCENTAGE OF COMPLEX WORDS']
        out_row['FOG INDEX'] = analysis['FOG INDEX']
        out_row['AVG NUMBER OF WORDS PER SENTENCE'] = analysis['AVG NUMBER OF WORDS PER SENTENCE']
        out_row['COMPLEX WORD COUNT'] = analysis['COMPLEX WORD COUNT']
        out_row['WORD COUNT'] = analysis['WORD COUNT']
        out_row['SYLLABLE PER WORD'] = analysis['SYLLABLE PER WORD']
        out_row['PERSONAL PRONOUNS'] = analysis['PERSONAL PRONOUNS']
        out_row['AVG WORD LENGTH'] = analysis['AVG WORD LENGTH']

        results.append(out_row)

    out_df = pd.DataFrame(results)
    # ensure column order: original input columns in original order
    ordered_cols = list(df_input.columns) + [
        'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE',
        'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX',
        'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 'WORD COUNT',
        'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'
    ]
    # filter to columns actually present
    ordered_cols = [c for c in ordered_cols if c in out_df.columns]
    out_df = out_df[ordered_cols]

    out_df.to_csv(output_csv, index=False)
    out_df.to_excel(output_xlsx, index=False)
    logging.info(f"Saved outputs to {output_csv} and {output_xlsx}")


if __name__ == '__main__':
    main()
