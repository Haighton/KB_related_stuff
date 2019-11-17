#!/usr/bin/env python3
"""Check if date in ALTO is same as METS/MODS of a dutch digitized newspaper.

Try to get publicationdate from the frontpage of a digitized newspaper by
searching de first ALTO file of an issue and searching for date patterns. The
best candidate is compared to the date in the corresponding METS/MODS XML file.

Start/input:
    $ python date_check_0_0_3.py "location/to/a/KB-digi-newspaper/batch_folder"

Method:
    - Find months with fuzzy string matching. Newspaper publicationdate is
      written as '24 januari 1984'.
    - Check if there are 2 int's before and 4 int's after the month name.

    - Scoring points to get best possible match:
        - Kernel Density Estimation.
          Location of date is always in the same spot in a specific title;
          "hotspot" - large density of points.
        - Convert VPOS value to a score.
          Publication dates are always in the header.
       (- K-means.
          Largest cluster is probably the publicationdate - aanname.
          Not really used anymore - KDE was more effective. Could use KDE to
          set initial cluster centroid - could be effective.)

Output:
    HTML file containing table with most likely candidates of having a
    wrong publicationdate with date snippets from the scan to check if it
    really is an error. Set global var `HTML_LOG_OUT` to a location, the script
    will create a folder with pattern
    `HTML_LOG_OUT/batch_id-datumcontrole/images`.

TODO:
    - Snippet is ACCESS hyperlink, mets_date is hyperlink naar METS. Issue ID
      ('filename') is link naar object map.
    - Info over batch en link in inleiding text boven tabel (uitgebreider?)
    - Check maybe density clustering instead of k-means and KDE?
    - batch_id in log naam.

"""

__author__ = "Thomas Haighton"
__contact__ = "thomas.haighton@kb.nl"
__date__ = "02-07-2019"
__version__ = "0.0.3"
__status__ = "Prototype"

import os
import sys
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lxml import etree
from PIL import Image

from fuzzywuzzy import fuzz
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import gaussian_kde

# Location to save html output.
# e.g. '/Users/haighton_imac/Desktop' or "F:\Desktop"
HTML_LOG_OUT = r"F:\Desktop"


def get_alto(path_batch):
    """Get first ALTO file of all objects in a batch.

    Only get first pages, thats where publicationdates are printed.
    files ending in _00001_alto.xml.

    Arguments:
        path_batch (str): Path to batch directory.

    Returns:
        List: Paths to all first ALTO files.
    """
    alto_files = []
    mets_files = []
    for dirpath, dirnames, filenames in os.walk(path_batch):
        for filename in filenames:
            if filename.endswith('_00001_alto.xml'):
                print('Searching files: {}'.format(filename),
                      end='\r', flush=True)
                alto_files.append(os.path.join(dirpath, filename))
            elif filename.endswith('_mets.xml'):
                mets_files.append(os.path.join(dirpath, filename))
    print('\n\nFound {} ALTO and {} METS files in {}.\n'.format(
          len(alto_files), len(mets_files), os.path.basename(path_batch)))
    return(alto_files, mets_files)


def get_alto_data(alto_file):
    """Extract //String/@CONTENT attributes from a ALTO XML file.

    Arguments:
        alto_file (str): Path to a ALTO file.

    Returns:
        list: Values (words) of each CONTENT attribute and a tuple with VPOS
        and HPOS.
    """
    alto_file_content = []
    with open(alto_file, 'rb') as alto:
        context = etree.iterparse(alto, events=('start', 'end'))
        for event, elem in context:
            if (event == 'end' and etree.QName(elem.tag)
               .localname == 'String'):
                alto_file_content.append([elem.get('CONTENT'),
                                         (int(elem.get('VPOS')),
                                          int(elem.get('HPOS')))])
    elem.clear()
    alto.close()
    return(alto_file_content)


def extract_mets_data(mets_files):
    r"""
    Extract publicationdate data from METS files.

    Arguments:
        mets_file (list): Paths to METS files.

    Returns:
        dict_mets_titles (dict): File ID and title {file_id: title}.
    """
    dict_mets_dates = {}
    filenames = []
    mets_dates = []
    mets_titles = []
    count = 0
    for mets_file in mets_files:
        count += 1
        print('Extracting data from METS: {} ({}/{}).'.format(
              os.path.basename(mets_file), count, len(mets_files)),
              end='\r', flush=True)
        np_titles = []
        np_dates = []
        with open(mets_file, 'rb') as mets:
            context = etree.iterparse(mets, events=('start', 'end'))
            for event, elem in context:
                if elem.tag == '{http://www.loc.gov/mods/v3}mods':
                    for elem_child in elem.getchildren():
                        if elem_child.tag == '{http://www.loc.gov/mods/v3}relatedItem':
                            for elem_child2 in elem_child.getchildren():

                                # Extract Newspaper title.
                                if elem_child2.tag == '{http://www.loc.gov/mods/v3}titleInfo':
                                    for elem_child3 in elem_child2.getchildren():
                                        if elem_child3.tag == '{http://www.loc.gov/mods/v3}title':
                                            np_titles.append(elem_child3.text)

                                # Extract Newspaper publication date.
                                elif elem_child2.tag == '{http://www.loc.gov/mods/v3}part':
                                    for elem_child3 in elem_child2.getchildren():
                                        if elem_child3.tag == '{http://www.loc.gov/mods/v3}date':
                                            np_dates.append(elem_child3.text)

        mets_titles.append(np_titles[0])
        mets_dates.append(np_dates[0])

        filenames.append(os.path.basename(mets_file).strip('_mets.xml'))
        elem.clear()

    dict_mets_dates = {'filename': filenames,
                       'mets_date': mets_dates,
                       'mets_title': mets_titles}
    return(dict_mets_dates)


def silhouette(df):
    r"""Determine optimal number of clusters using silhouette for k-means.

    https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py

    Arguments:
        df (df): [filename, alto_date, VPOS, HPOS]

    Returns:
        int: Optimal number of clusters for k-means function.
    """
    X = df.iloc[:, [2, 3]]
    range_n_clusters = [2, 3, 4, 5, 6]
    sil_score = {}
    for n_clusters in range_n_clusters:
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the
        # formed clusters.
        silhouette_avg = silhouette_score(X, cluster_labels)
        sil_score[n_clusters] = silhouette_avg
    # Return nr of clusters with highest silhouette score.
    return(max(sil_score, key=sil_score.get))


def clustering(df, nr_clusters):
    """K-means clustering to determine correct date from potential candidates.

    Choose best option by if its part of the largest cluster.

    Uniformity of clusters - they all have stable VPOS HPOS t.o.v. centroid.?

    Arguments:
        df (df): [filename, alto_date, vpos, hpos,
                  kde_score, vpos_score, score]
    """
    km = KMeans(n_clusters=nr_clusters,
                random_state=0).fit(df[['VPOS', 'HPOS']])
    result = km.labels_
    df['label'] = result

    # centroids = km.cluster_centers_
    # Get biggest group.
    # print(df.sort_values(by=['label']))
    # print(centroids)
    # biggest_grp = df_group.reset_index().iloc[0, 0]
    # test = df[df['label'] == biggest_grp]
    # print(test)

    return(df)


def vpos_score(df):
    """Convert VPOS into score 0-1.

    Publicationdate is always shown at top of the page. VPOS min and max are
    scaled to 0.0-1.0 a d used as a score.

    Arguments:
        df (df): [filename, alto_date, vpos, hpos, kde_score]]

    Returns:
        df: [filename, alto_date, vpos, hpos, kde_score, vpos_score]]
    """
    df['vpos_score'] = np.round(1 - (df['VPOS']-df['VPOS'].min()) /
                                (df['VPOS'].max()-df['VPOS'].min()), 2)
    return(df)


def kde_gaussian(df):
    """Kernel Density Estimation using Gaussian kernels.

    Find probability density function.
    """
    x = df['HPOS']
    y = df['VPOS']

    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Scale gaussian_kde to 0-1 and add to df.
    z = np.round((z - min(z)) / (max(z) - min(z)), 2)
    df['kde_score'] = z
    return(df)


def compare_dates(df):
    """Compare alto_date with mets_date."""
    # [yyyy, mm, dd]
    a_date = df['alto_date'].str.split('-')
    m_date = df['mets_date'].str.split('-')

    # Remove lead 0 from single digit month and day values.
    def lead_zero(dates):
        remove_zeros = []
        for ad in dates:
            entry = []
            for adf in ad:
                entry.append(adf.lstrip('0'))
            remove_zeros.append(entry)
        return(remove_zeros)
    a_date = lead_zero(a_date)
    m_date = lead_zero(m_date)

    # Compare day, month and year seperate (day should have bigger weight?).
    dis_score = []
    for i in range(0, len(m_date)):
        year_diff = abs(int(a_date[i][0]) - int(m_date[i][0]))
        month_diff = abs(int(a_date[i][1]) - int(m_date[i][1]))
        day_diff = abs(int(a_date[i][2]) - int(m_date[i][2]))
        dis_score.append(year_diff + month_diff + day_diff)

    df['distance_score'] = dis_score
    return(df)


def plot_fig(df_errors, path_batch, current_title, log_path):
    """Plot df_errors VPOS HPOS."""
    # Path to first access of first file in df.
    bgfile = df_errors.iloc[0, 0]
    acpath = os.path.join(os.path.join(
                          os.path.join(path_batch, bgfile), 'access'),
                          bgfile + '_00001_access.jp2')
    img = plt.imread(acpath)

    fig, ax = plt.subplots()

    # Scatter plot.
    plt.scatter(df_oldy['HPOS'], df_oldy['VPOS'], c=df_oldy['score'],
                s=50, cmap='inferno', alpha=0.4)

    # Title and x-y labels
    plt.title("All Coordinates of Dates Found in ALTO's")
    plt.xlabel('HPOS')
    plt.ylabel('VPOS')

    # Invert y-axis, ALTO top-left is (0, 0).
    plt.gca().invert_yaxis()

    # Colorbar legend.
    cbar = plt.colorbar()
    cbar.set_label('Score', rotation=270)
    cbar.ax.tick_params(size=0)

    # Border around plot.
    ax.spines['bottom'].set_color('0.5')
    ax.spines['top'].set_color('0.5')
    ax.spines['right'].set_color('0.5')
    ax.spines['left'].set_color('0.5')

    # Background image.
    ax.imshow(img, alpha=0.7)
    current_title_ = ''.join(current_title.split())[:6]
    fig_filename = 'fig_' + current_title_ + '.png'
    plt.savefig(os.path.join(
                os.path.join(log_path, 'images'),
                fig_filename))
    # plt.show()


def generate_html_log(df_html_table_er, path_batch, current_title, log_path):
    """Generate a HTML table with all the candidates.

    Arguments:
        df_html_table_er (df): All the top candidates only.
    """
    df_html_table_er = df_html_table_er.reset_index()
    df_html_table_er = df_errors[['filename', 'alto_date', 'mets_date',
                                  'VPOS', 'HPOS']]

    # Generate access path for hypertext.
    access_paths = []
    html_access_paths = []
    file_ids = []
    for file_id in df_errors['filename']:
        access_path = os.path.join(os.path.join(
                                   os.path.join(path_batch, file_id),
                                   'access'),
                                   file_id + '_00001_access.jp2')
        html_access_path = '<a target="_blank" href="' + access_path + '">' + \
                           file_id + '_00001_access.jp2</a>'
        access_paths.append(access_path)
        file_ids.append(file_id)
        html_access_paths.append(html_access_path)
    df_access = pd.DataFrame({'filename': file_ids,
                              'access_path': access_paths,
                              'html_access_path': html_access_paths},
                             columns=['filename',
                                      'access_path',
                                      'html_access_path'])
    df_wpath = df_html_table_er.merge(df_access, on='filename')

    # Cut-out of date coordinates (VPOS, HPOS) to display in html table.
    # Make a sample image from access file.
    df_image_cut = df_wpath[['filename', 'access_path', 'VPOS', 'HPOS']]

    fileid = df_image_cut['filename'].to_list()
    image_p = df_image_cut['access_path'].to_list()
    vpos = df_image_cut['VPOS'].to_list()
    hpos = df_image_cut['HPOS'].to_list()

    fnames = []
    thumb_paths = []
    for i in range(0, len(image_p)):
        print('Creating date thumbnail: {}.'.format(
              os.path.basename(image_p[i])), end='\r', flush=True)
        # Cut sample images at VPOS HPOS.
        img = Image.open(image_p[i])
        width, height = img.size

        # Location and Size of date thumbnails - zoeken naar optimale grootte
        # Alle informatie erop, en niet teveel extra shizz als dag naam.
        # Mogelijk in latere versie HPOS van dag en jaar meenemen.
        crop_img = img.crop((hpos[i] - 140, vpos[i] - 20,
                             hpos[i] + 700, vpos[i] + 80))
        baseheight = 30
        try:
            hpercent = (baseheight / float(crop_img.size[1]))
            wsize = int((float(crop_img.size[0]) * float(hpercent)))
            crop_img = crop_img.resize((wsize, baseheight), Image.ANTIALIAS)
            impath_abs = os.path.join(os.path.join(log_path, 'images'),
                                      os.path.basename(image_p[i]).rstrip(
                                      'access.jp2') + 'date.jpg')
            impath_rel = os.path.join('images',
                                      os.path.basename(image_p[i]).rstrip(
                                                       'access.jp2') +
                                      'date.jpg')

            thumb_paths.append('<img src="' + impath_rel + '" alt="test">')
            fnames.append(fileid[i])

            crop_img.save(impath_abs, 'JPEG', quality=10)
        except ZeroDivisionError:
            continue

    data_img = {'filename': fnames, 'thumb_paths': thumb_paths}
    df_img_paths = pd.DataFrame(data_img, columns=['filename', 'thumb_paths'])

    pd.set_option('display.max_colwidth', -1)
    df_wpath_ = df_wpath[['filename', 'alto_date',
                          'mets_date', 'html_access_path']]
    df_wpath_ = df_wpath_.merge(df_img_paths, on='filename')

    # Change order of columns.
    col_html = ['filename', 'mets_date', 'thumb_paths', 'html_access_path']
    df_wpath_ = df_wpath_.loc[:, col_html]

    # Rename columns (intern gebruikte termen / betere termen).
    df_wpath_ = df_wpath_.rename(columns={
                                 'filename': 'Issue ID',
                                 'mets_date': 'Publicatiedatum Metadata',
                                 'thumb_paths': 'Datum scan',
                                 'html_access_path': 'Access bestand openen'})
    html_table_er = df_wpath_.to_html(escape=False)

    html_title = ('<h2>Datum discrepanties in '
                  + os.path.basename(path_batch) + '</h2><h3>Krant: '
                  + current_title + ' </h3>')
    # HTML snippets for logfile.
    html_start = """<!DOCTYPE html>
<html>
<head>
    <title>Log Publicatiedatum controle ALTO-METS</title>
    <style>
        table {
            font-family: "Trebuchet MS", Arial, Helvetica, sans-serif;
            border-collapse: collapse;
            width: 100%;
        }
        td, th {
            border: 1px solid #ddd;
            padding: 8px;
        }
        tr:nth-child(even){background-color: #f2f2f2;}
        tr:hover {background-color: #ddd;}
        th {
            padding-top: 12px;
            padding-bottom: 12px;
            text-align: left;
            background-color: #4CAF50;
            color: white;
        }
    </style>
</head>
<body>
    <h1>Publicatiedatum controle</h1>
    """
    # Shorten name (to 6 - is semi-random) bc of errors w/ long names+spaces.
    current_title_ = ''.join(current_title.split())[:6]

    # Save location of matplot png.
    html_figure = '<img src="images/fig_' + current_title_ + '.png">'

    html_end = '</body></html>'

    # Put HTML pieces together.
    html_result = (html_start
                   + html_title
                   + html_table_er
                   + html_figure
                   + html_end)

    # Save HTML as logfile.html
    log_filename = 'logfile_' + current_title_ + '.html'
    with open(os.path.join(log_path, log_filename),
              'w') as html_out:
        html_out.write(html_result)
        html_out.close()


if __name__ == '__main__':
    # Get paths to ALTO en METS files in given batch.
    alto_files, mets_files = get_alto(sys.argv[1])

    dict_fuzz_score = {}
    filenames = []
    alto_dates = []
    content_x = []
    content_y = []
    months = {'januari': '01',
              'februari': '02',
              'maart': '03',
              'april': '04',
              'mei': '05',
              'juni': '06',
              'juli': '07',
              'augustus': '08',
              'september': '09',
              'oktober': '10',
              'november': '11',
              'december': '12'}
    count = 0
    for alto_file in alto_files:
        count += 1
        print('Processing {} ({}/{}).'.format(os.path.basename(alto_file),
              count, len(alto_files)), end='\r', flush=True)
        word_count = 0
        alto_content = get_alto_data(alto_file)
        for alto_word in alto_content:
            word_count += 1
            # Ignore problematic words.
            if alto_word[0].lower() not in ['maar']:
                for month in months.keys():
                    # Convert to lowercase and fuzzy match with month names.
                    fuzz_score = fuzz.ratio(alto_word[0].lower(), month)
                    if fuzz_score > 80:
                        try:  # Remove punctuation of previous and next word.
                            prev_content = re.sub(r'[^\w\s]', '',
                                                  alto_content[word_count-2][0])
                            next_content = re.sub(r'[^\w\s]', '',
                                                  alto_content[word_count][0])
                            # If previous and next word are int
                            # and of limited length.
                            # TODO: No solve for 'i986' year - remove int cast.
                            if (int(prev_content) and len(prev_content) < 3 and
                               int(next_content) and len(next_content) < 5):
                                filenames.append(os.path.basename(alto_file)
                                                 .rstrip('_alto_00001.xml'))

                                # Date format we are going to use: yyyy-mm-dd
                                alto_dates.append(next_content +
                                                  '-' + months[month] +
                                                  '-' + prev_content.zfill(2))
                                content_x.append(alto_word[1][0])
                                content_y.append(alto_word[1][1])
                        except (ValueError, IndexError):
                            continue
    Data = {'filename': filenames, 'alto_date': alto_dates,
            'VPOS': content_x, 'HPOS': content_y}
    df = pd.DataFrame(Data, columns=['filename', 'alto_date', 'VPOS', 'HPOS'])

    # Extract METS data ['filename', 'mets_date', 'mets_title']
    dict_mets_dates = extract_mets_data(mets_files)

    # Import METS data, adds ['filename', 'mets_date'] to df.
    df_mets = pd.DataFrame(dict_mets_dates,
                           columns=['filename', 'mets_date', 'mets_title'])

    # Each newspaper need a seperate analysis.
    newspaper_titles = list(set(dict_mets_dates['mets_title']))
    for j in range(0, len(newspaper_titles)):

        # Change name of current_title - no spaces and 6 chars (got errors).
        current_title = newspaper_titles[j]

        # SCORES (KDE, VPOS scaling, K-means)
        # Each function calculates a score between 0-1. df['score'] has the
        # mean of these values and used as final score. Only a final score of
        # 80 and above are kept as potential candidates.
        if len(df) > 0:

            # Create folders for logfile and images.
            batch_id = os.path.basename(sys.argv[1])
            log_folder = os.path.join(HTML_LOG_OUT, batch_id + '-datumcontrole')
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)
                os.makedirs(os.path.join(log_folder, 'images'))
            elif (os.path.exists(log_folder) and not
                  os.path.exists(os.path.join(log_folder, 'images'))):
                os.makedirs(os.path.join(log_folder, 'images'))

            # Add METS data to main df.
            df = pd.merge(df, df_mets, on='filename')

            # DF only contains issue id's of current title.
            df = df[df['mets_title'] == newspaper_titles[j]]
            if len(df) > 0:
                # Kernel Density Estimation, adds ['kde_score'] to df.
                df = kde_gaussian(df)

                # VPOS score, adds ['vpos_score'] to df.
                df = vpos_score(df)

                # K-Means with silhouette, adds ['label'] to df.
                df = clustering(df, silhouette(df))

                # Calculate mean of scores, adds ['score'] to df.
                col = df.loc[:, 'kde_score':'vpos_score']
                df['score'] = np.round(col.mean(axis=1), 2)

                # Using this for plotting later.
                df_oldy = df

                # Only keep rows with score > 0.80.
                df = df[df['score'] > 0.8]

                # TODO: Take care of multiple candidates
                #       i.e. duplicate 'filename'!

                print('\n\nFound dates for {} pages.'.format(len(df)))
                alto_fnames = [os.path.basename(alto_path)
                               .rstrip('_00001_alto.xml')
                               for alto_path in alto_files]
                no_pd = set(alto_fnames) - set(df['filename'])
                print("Didn't find a candidate for {} file(s) ({}%).\n".format(
                    len(no_pd), np.round(len(no_pd)/len(alto_files)*100.), 1))

                # TODO: Write ALTO filenames without publicationdate to txt.

                # Put DATE cols next to each other.
                cols = ['filename', 'VPOS', 'HPOS', 'kde_score', 'vpos_score',
                        'score', 'label', 'alto_date', 'mets_date']
                df = df.loc[:, cols]

                # Compare alto_dates with mets_dates.
                df = compare_dates(df)

                # The ones which have the most potential to being a real error
                # is having a low distance_score (1-5), 0 being a exact match
                # in publicationdates. May change to use day in score only
                # (year/month not really important).
                df_errors = df[(df['distance_score'] > 0) &
                               (df['distance_score'] < 5)]
                df_match = df[df['distance_score'] == 0]

                # Create plot, save in 'images/fig.png'.
                if len(df_errors) > 0:
                    plot_fig(df_errors, sys.argv[1], current_title, log_folder)

                    # Create a fancy pants HTML output logfile.
                    generate_html_log(df_errors, sys.argv[1],
                                      current_title, log_folder)
                    print(df_errors)
                else:
                    print('\nFound {} errors in {}!'.format(
                          len(df_errors), current_title))

        else:
            print('No errors found in {}')
            print(df)
    print('\nDONE!')
