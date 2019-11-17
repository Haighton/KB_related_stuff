#!/usr/bin/env python3
"""KB Hackathon 2019.

ONDERZOEKSVRAAG
Kunnen we een beeld krijgen van het aantal freelancers dat heeft meegewerkt
aan de kranten op Delpher waarop nog auteursrecht rust (gepubliceerd in 1879
en daarna)?

HINT
Freelancers zijn te herkennen aan hun eigen naam die bij hun bijdrage
(artikel, foto, illustratie, etc.) vermeld staat. Waren ze in dienst van de
uitgever, dan werd er meestal geen eigen naam vermeld, maar begon een artikel
bijvoorbeeld met: ‘van een onzer redacteuren’.

METHODE
Medewerker:
    - Fuzzy string match specific sentences/words (e.g. 'van onze redactie').

Freelancer:
    - Named entity recognition om alle name uit de artikel tekst te halen.
    - Het woord voor de naam is "door".
    - Artikel eindigt met [PER][LOC].
    - Staat aan het begin/eind van het artikel?

USAGE
$python KBHackathon2019.py "path/to/dateset"

UITKOMST
- DF: [id, text_filename, newspaper_title, pub_date, medewerker/Freelancer/unk]
- Pie chart with results for for whole dataset.
- Bar charts? for each individual newspaper over time?

TODO
- Use other newspaper titles than 'Telegraaf' and find rules.
- Nieuws van persbureaus (als Reuters).
- Foto: door [fotograaf naam]
- Use ALTO to check STYLEREFS to score certainty (author differs from body).
"""
import sys
import os
import re

from lxml import etree
from fuzzywuzzy import fuzz
import pandas as pd
import spacy


def get_files(path_data):
    """Locate all text and mpeg21 files in dataset.

    Arguments:
        path_data (str): Path to dataset ("/Volumes/Elements/hackathon2019").

    Returns:
        list: Paths to text files.
        list: Paths to mpeg21 files.
    """
    text_files = []
    mpeg_files = []
    for dirpath, dirnames, filenames in os.walk(path_data):
        for filename in filenames:
            if filename.endswith('_ocr.xml'):
                print('Getting files: {}'.format(filename),
                      end='\r', flush=True)
                text_files.append(os.path.join(dirpath, filename))
            elif filename.endswith('_didl.xml'):
                mpeg_files.append(os.path.join(dirpath, filename))
    print('\nFound {} text files.'.format(len(text_files)))
    print('Found {} mpeg21 files.\n'.format(len(mpeg_files)))
    return(text_files, mpeg_files)


def extract_text_xml(text_files):
    """Extract data from text XML.

    We are going to join the title tag with the rest of the text - separated
    by a '.', because a lot of times the important data is found in the title
    tag.

    Arguments:
        text_files (list): Paths to full text files.

    Returns:
        df: [filename, text, id]
    """
    dict_text = {}
    for text_file in text_files:
        p_all = []
        print('Extracting text data: {}'.format(os.path.basename(text_file)),
              end='\r', flush=True)
        with open(text_file, 'rb') as text:
            context = etree.iterparse(text, events=('start', 'end'))
            for event, elem in context:
                if event == 'end' and elem.tag == 'title':
                    title = elem.text
                elif event == 'end' and elem.tag == 'p':
                    p_all.append(elem.text)
        try:
            all_text = title + '. ' + ''.join(p_all)
            dict_text[os.path.basename(text_file)] = all_text
        except TypeError:
            continue
    df = pd.DataFrame(list(dict_text.items()), columns=['filename', 'text'])
    df['id'] = df['filename'].str[:-14]
    return(df)


def extract_mpeg_xml(mpeg_files):
    """Extract data from mpeg21 XML.

    Arguments:
        mpeg_files (list): Paths to mpeg21 files.

    Returns:
        df_mpeg (df): [id, newspaper, publicationdate]
    """
    mpeg_titles = []
    mpeg_dates = []
    mpeg_ids = []
    dict_mpeg = {}
    for mpeg_file in mpeg_files:
        print('Extracting mpeg data: {}'.format(os.path.basename(mpeg_file)),
              end='\r', flush=True)
        nd_titles = []
        nd_dates = []
        with open(mpeg_file, 'rb') as mpeg:
            context = etree.iterparse(mpeg, events=('start', 'end'))
            for event, elem in context:
                if (event == 'end' and
                   elem.tag == '{info:srw/schema/1/dc-v1.1}dcx'):
                    for elem_child in elem.getchildren():
                        if elem_child.tag == '{http://purl.org/dc/elements/1.1/}title':
                            nd_titles.append(elem_child.text)
                        elif elem_child.tag == '{http://purl.org/dc/elements/1.1/}date':
                            nd_dates.append(elem_child.text)
        mpeg_titles.append(nd_titles[0])
        mpeg_dates.append(nd_dates[0])
        mpeg_ids.append(os.path.basename(mpeg_file)[:-9])
    
    dict_mpeg = {'id': mpeg_ids,
                 'newspaper': mpeg_titles,
                 'publicationdate': mpeg_dates}
    df_mpeg = pd.DataFrame(dict_mpeg,
                           columns=['id', 'newspaper', 'publicationdate'])
    return(df_mpeg)


def fuzzy_match(df):
    """Fuzzy match specific sentences.

    Text which start with 'van onze correspondent' and its variations.

    Arguments:
        df (df): Main DataFrame. 
   
    Returns:
        df_mw (df): [filename, medewerker]
    """
    all_text = df['text'].values.tolist()
    all_fnames = df['filename'].values.tolist()
    dict_medewerker = {}
    for i in range(0, len(all_fnames)):
        print('Performing fuzzy match: {}'.format(all_fnames[i]), 
              end='\r', flush=True)
        
        medewerker = ['van onze', 'van een onzer']
        
        freelancers = ['TON DE ZEEUW', 'FEL EN GAMMIDGE', 'ROBERT KROON',
                       'BERRY ZAND SCHOLTEN', 'JAN DE DEUGD', 'HANS WOUDSTRA',
                       'RON COUWEHOVEN', 'CO BERKENBOSCH', 'MIA SNELDER',
                       'KIRSTEN COENRADIE', 'KEES ROOS', 'HENK TEN BERGE']
                       
        
        # TODO: Gebruik NLTK om in zinnen op te delen i.p.v. split('.').
        # Alleen de eerste 5 zinnen gebruiken.
        for sen in all_text[i].split('.')[:5]:
            sen = re.sub('[*•■;,\'\-\"\:><\/~»]', '', sen)
            
            # Only check sentences longer than 10 chars.
            if len(sen) > 10:
                for choice in medewerker:
                    fuzzscore = fuzz.token_set_ratio(sen, choice)

                    # 98 seems the work okay as threshold.
                    if fuzzscore > 98:
                        dict_medewerker[all_fnames[i]] = 'medewerker'
                    else:
                        for freelancer in freelancers:
                            fuzzfl = fuzz.token_set_ratio(sen, freelancer)
                            if fuzzfl > 98:
                                dict_medewerker[all_fnames[i]] = 'freelancer'

    df_mw = pd.DataFrame(list(dict_medewerker.items()),
                         columns=['filename', 'status'])
    return(df_mw)


def ner(df):
    """Find freelance author of newspaper article.
    
    Get personal names from the start of article using named enity recognition
    and check if the word in front of the name is 'door'.

    Arguments:
        df (df): Main DataFrame.

    Returns:
        df_fl (df): [filename, status]
    """
    nlp = spacy.load("nl_core_news_sm")
    all_text = df['text'].values.tolist()
    all_fnames = df['filename'].values.tolist()
    freelance = {}
    count = 0
    for i in range(0, len(all_fnames)):
        count += 1
        print('Performing named entity recognition: {} [{}/{}]'.format(
              all_fnames[i], count, len(all_fnames)), end='\r', flush=True)
        doc = nlp(all_text[i])
        for ent in doc.ents:
            # TODO: Fuzzy match "door"?
            # De Telegraaf: Start van artikel 'door [auteur]'
            if (ent.label_ == "PER" 
               and all_text[i][ent.start_char-5:ent.start_char-1].lower() == 'door'
               and ent.start_char < 150):
                freelance[all_fnames[i]] = 'freelancer'
    
    df_fl = pd.DataFrame(list(freelance.items()),
                         columns=['filename', 'status'])
    return(df_fl)


if __name__ == '__main__':
    text_files, mpeg_files = get_files(sys.argv[1])
    df = extract_text_xml(text_files)
    df_mpeg = extract_mpeg_xml(mpeg_files)
    df = pd.merge(df, df_mpeg, on='id')
    df_mw = fuzzy_match(df)
    df = df.merge(df_mw, on='filename', how='left')
    df_fl = ner(df)
    
    df_fl.set_index('filename', inplace=True)
    df.set_index('filename', inplace=True)
    df = df.combine_first(df_fl)
    print(df)
    
    # Export df as CSV.
    df.to_csv('/Users/haighton_macbook/Desktop/Projects/KB/KBHackathon2019/data/kbhack_final.csv')
