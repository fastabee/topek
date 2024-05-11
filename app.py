from flask import Flask, render_template, request, flash
import tweepy
import re
import string
import csv
import pickle
import os
from os.path import join, dirname, realpath
import pandas as pd
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from googletrans import Translator
from textblob import TextBlob
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,precision_score,recall_score
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from PIL import Image
import urllib.request
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import math

nltk.download('punkt')
nltk.download('stopwords')

hasil_preprocessing = []
hasil_labeling = []

app = Flask(__name__)
app.config['SECRET_KEY'] = 'farez'

UPLOAD_FOLDER = 'static/files'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


dataNormalisasi = pd.read_csv('stati\kamus_singkatan.csv', sep=',')

# ubah data normalisasi menjadi dictionary
dictNormalisasi = dict(zip(dataNormalisasi['singkatan'], dataNormalisasi['asli']))

hasil_preprocessing = []

def normalisasi_kata(teks, dict_normalisasi):
    teks = teks.split()
    hasil = []
    for kata in teks:
        if kata.lower() in dict_normalisasi:
            hasil.append(dict_normalisasi[kata.lower()])
        else:
            hasil.append(kata)
    return " ".join(hasil)

# def preprocessing_twitter():
    dataNormalisasi = pd.read_csv('stati\kamus_singkatan.csv', sep=',')
    dictNormalisasi = dict(zip(dataNormalisasi['singkatan'], dataNormalisasi['asli']))

    hasil_preprocessing.clear()

    with open("static/files/Data Scraping.csv", "r", encoding='utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        
        # Skip the header row
        next(readCSV)

        for row in readCSV:
            clean = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", row[2]).split())
            clean = re.sub("\d+", "", clean)
            clean = re.sub(r"\b[a-zA-Z]\b", "", clean)
            clean = re.sub('\s+', ' ', clean)
            clean = clean.translate(clean.maketrans("", "", string.punctuation))
            casefold = clean.casefold()
            normakisais = normalisasi_kata(casefold, dictNormalisasi)
            tokenizing = nltk.tokenize.word_tokenize(normakisais)
            stop_word_remover = StopWordRemoverFactory().create_stop_word_remover()
            stop_wr = nltk.tokenize.word_tokenize(stop_word_remover.remove(normakisais))
            kalimat = ' '.join(stop_wr)
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            stemming = stemmer.stem(kalimat)

            tweets = [row[0], row[1], row[2], clean, casefold, normakisais, tokenizing, stop_wr, stemming]
            hasil_preprocessing.append(tweets)

            # writer.writerow(tweets)
            flash('Preprocessing Berhasil', 'preprocessing_data')
def preprocessing_twitter():
    # Membuat File CSV
    
    # d = pd.read_csv(csv_path)
    # writer = csv.writer(d)
    # file = open('E:\Archieve File\Analisis-Sentimen-Twitter-Flask\static/files/Data Preprocessing.csv', 'w', newline='', encoding='utf-8')
    # writer = csv.writer(file)
    csv_path = os.path.join(app.root_path, 'static', 'files', 'Data Preprocessing.csv')
    writer = csv.writer(open(csv_path, 'w', encoding='utf-8', newline=''))
    with open("static/files/Data Scraping.csv", "r", encoding='utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        hasil_labeling.clear()
        for row in readCSV:
            # proses cleansing
            # remove mention, link, hashtag
            clean = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", row[2]).split())
            # remove number
            clean = re.sub("\d+", "", clean)
            # remove single char
            clean = re.sub(r"\b[a-zA-Z]\b", "", clean)
            # remove multiple whitespace menjadi satu spasi
            clean = re.sub('\s+', ' ', clean)
            # remove punctuation (emoji)
            clean = clean.translate(clean.maketrans("", "", string.punctuation))

            # proses casefolding
            casefold = clean.casefold()

            # normalisasi kata
            normakisais = normalisasi_kata(casefold, dictNormalisasi)

            # proses tokenizing
            # tokenizing = nltk.tokenize.word_tokenize(casefold)
            tokenizing = nltk.tokenize.word_tokenize(normakisais)

            # proses stopword
            # mengambil data stop word dari library
            stop_factory = StopWordRemoverFactory().get_stop_words()
            # menambah stopword sendiri
            more_stop_word = ['&amp', 'ad', 'ada', 'ae', 'ah','ai',
                              'aja', 'ajar', 'ajar', 'amp', 'apa', 'aya', 'bab', 'bajo', 
                              'bar', 'bbrp', 'beda', 'begini', 'bgmn', 'bgt', 'bhw', 'biar', 'bikin', 
                              'bilang', 'bkh', 'bkn', 'bln', 'bnyk', 'brt', 'buah', 'cc', 'cc', 
                              'ckp', 'com', 'cuy', 'd', 'dab', 'dah', 'dan', 'dg', 'dgn', 'di', 
                              'dih', 'dlm', 'dm', 'dpo', 'dr', 'dr', 'dri', 'duga', 'duh', 'enth', 
                              'er', 'et', 'ga', 'gak', 'gal', 'gin', 'gitu', 'gk', 'gmn', 'gs', 
                              'gt', 'gue', 'gw', 'hah', 'hallo', 'halo', 'hehe', 'hello', 'hha', 
                              'hrs', 'https', 'ia', 'iii', 'in', 'ini', 'iw', 'jadi', 'jadi', 'jangn', 
                              'jd', 'jg', 'jgn', 'jls', 'kak', 'kali', 'kalo', 'kan', 'kch', 'ke', 
                              'kena', 'ket', 'kl', 'kll', 'klo', 'km', 'kmrn', 'knp', 'kok', 'kpd', 
                              'krn', 'kui', 'lagi', 'lah', 'lahh', 'lalu', 'lbh', 'lewat', 'loh', 'lu', 
                              'mah', 'mau', 'min', 'mlkukan', 'mls', 'mnw', 'mrk', 'n', 'nan', 'ni', 
                              'nih', 'no', 'nti', 'ntt', 'ny', 'nya', 'nyg', 'oleh', 'ono', 'ooooo', 
                              'op', 'org', 'pen', 'pk', 'pun', 'qq', 'rd', 'rt', 'sama', 'sbg', 'sdh', 
                              'sdrhn', 'segera', 'sgt', 'si', 'si', 'sih', 'sj', 'so', 'sy', 't', 'tak', 
                              'tak', 'tara', 'tau', 'td', 'tdk', 'tdk', 'thd', 'thd', 'thn', 'tindkn', 
                              'tkt', 'tp', 'tsb', 'ttg', 'ttp', 'tuh', 'tv', 'u', 'upa', 'utk', 'uyu', 
                              'viral', 'vm', 'wae', 'wah', 'wb', 'wes', 'wk', 'wkwk', 'wkwkwk', 'wn', 
                              'woiii', 'xxxx', 'ya', 'yaa', 'yah', 'ybs', 'ye', 'yg', 'ykm']
            # menggabungkan stopword library + milik sendiri
            data = stop_factory + more_stop_word

            dictionary = ArrayDictionary(data)
            str = StopWordRemover(dictionary)
            # stop_wr = nltk.tokenize.word_tokenize(str.remove(casefold))
            stop_wr = nltk.tokenize.word_tokenize(str.remove(normakisais))

            # proses stemming
            kalimat = ' '.join(stop_wr)
            factory = StemmerFactory()
            # mamanggil fungsi stemming
            stemmer = factory.create_stemmer()
            stemming = stemmer.stem(kalimat)

            tweets = [row[0], row[1], row[2], clean, casefold,normakisais, tokenizing, stop_wr, stemming]
            hasil_preprocessing.append(tweets)

            writer.writerow(tweets)
            flash('Preprocessing Berhasil', 'preprocessing_data')


def labeling_twitter():
    data_preprocessing_csv_path = os.path.join(app.root_path, 'static', 'files', 'Data Preprocessing.csv')
    labeling_csv_path = os.path.join(app.root_path, 'static', 'files', 'Data Labeling.csv')

    translator = Translator()

    try:
        with open(data_preprocessing_csv_path, "r", encoding='utf-8') as preprocessing_file, \
                open(labeling_csv_path, 'w', encoding='utf-8', newline='') as labeling_file:

            csv_reader = csv.reader(preprocessing_file, delimiter=',')
            csv_writer = csv.writer(labeling_file)

            hasil_labeling.clear()

            for row in csv_reader:
                tweet = {}
                try:
                    value = translator.translate(row[8], dest='en')  # Changed index from 7 to 2, jika diganti 2 maka hasil akurasi 58%
                except Exception as e:
                    print(f"Terjadi kesalahan {e}", flush=True)

                terjemahan = value.text
                data_label = TextBlob(terjemahan)

                if data_label.sentiment.polarity > 0.0:
                    tweet['sentiment'] = "Positif"
                elif data_label.sentiment.polarity == 0.0:
                    tweet['sentiment'] = "Netral"
                else:
                    tweet['sentiment'] = "Negatif"

                labeling = tweet['sentiment']
                tweets = [row[1], row[8], labeling]  # Changed index from 7 to 2
                hasil_labeling.append(tweets)

                csv_writer.writerow(tweets)

        flash('Labeling Berhasil', 'labeling_data')

    except Exception as e:
        print(f"Terjadi kesalahan {e}", flush=True)
        # 


#Klasifikasi

# Membuat variabel df
df = None
df2 = None

# menentukan akurasi 0
akurasi = 0

def proses_klasifikasi():
    global df
    global df2
    global akurasi
    tweet = []
    y = []

    with open("static/files/Data Labeling.csv", encoding='utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            tweet.append(row[1])
            y.append(row[2])

    vectorizer = TfidfVectorizer()
    vectorizer.fit(tweet)
    # tfidf = vectorizer.fit_transform(X_train)
    x = vectorizer.transform(tweet)

# tfidf
# Calculate term frequencies
    # term_frequencies = np.sum(x.toarray(), axis=0)
# Calculate document frequencies
    # document_frequencies = np.sum(x.toarray() != 0, axis=0)
# Calculate inverse document frequencies
    # inverse_document_frequencies = np.log(x.shape[0] / (1 + document_frequencies))
# Calculate TF-IDF values
    # tf_idf_values = term_frequencies * inverse_document_frequencies
# Convert TF-IDF matrix to DataFrame
    # tfidf_df = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())
# end tfidf


    # split data training dan testing 80:20
    # x_train, x_test, y_train, y_test = train_test_split(
    #     x, y, test_size=0.2, random_state=42)

    #split data training dan testing 90:10
    # x_train, x_test, y_train, y_test = train_test_split(
    #     x, y, test_size=0.1, random_state=42)

    #split data training dan testing 60:40
    # x_train, x_test, y_train, y_test = train_test_split(
    #     x, y, test_size=0.4, random_state=42)
    
    #split data training dan testing 70:30
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42)
    
    # metode NB
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    
    predict = clf.predict(x_test)
    report = classification_report(y_test, predict, output_dict=True)
    
    # simpan ke csv
    clsf_report = pd.DataFrame(report).transpose()
    clsf_report.to_csv(
        'static/files/Data Klasifikasi.csv', index=True)

    pickle.dump(vectorizer, open('static/files/vec.pkl', 'wb'))
    pickle.dump(x, open('static/files/tfidf.pkl', 'wb'))
    pickle.dump(clf, open('static/files/model.pkl', 'wb'))

    #Confusion Matrix
    unique_label = np.unique([y_test, predict])
    cmtx = pd.DataFrame(
        confusion_matrix(y_test, predict, labels=unique_label),
        index=['pred:{:}'.format(x) for x in unique_label],
        columns=['true:{:}'.format(x) for x in unique_label]
    )   

    cmtx.to_csv(
        'static/files/Data Confusion Matrix.csv', index=True)

    df = pd.read_csv(
        'static/files/Data Confusion Matrix.csv', sep=",")
    df.rename(columns={'Unnamed: 0': ''}, inplace=True)

    df2 = pd.read_csv(
        'static/files/Data Klasifikasi.csv', sep=",")
    df2.rename(columns={'Unnamed: 0': ''}, inplace=True)

    akurasi = round(accuracy_score(y_test, predict) * 100, 2)

    kalimat = ""

    for i in tweet:
        s = ("".join(i))
        kalimat += s

    urllib.request.urlretrieve(
        "https://firebasestorage.googleapis.com/v0/b/sentimen-97d49.appspot.com/o/Circle-icon.png?alt=media&token=b9647ca7-dfdb-46cd-80a9-cfcaa45a1ee4", 'circle.png')
    mask = np.array(Image.open("circle.png"))
    wordcloud = WordCloud(width=1600, height=800,
                          max_font_size=200, background_color='white', mask=mask)
    wordcloud.generate(kalimat)
    plt.figure(figsize=(12, 10))

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.savefig('static/files/wordcloud.png')


    # Pie chart
    counter = dict((i, y.count(i)) for i in y)
    isPositive = 'Positif' in counter.keys()
    isNegative = 'Negatif' in counter.keys()
    isNeutral = 'Netral' in counter.keys()

    positif = counter["Positif"] if isPositive == True else 0
    negatif = counter["Negatif"] if isNegative == True else 0
    netral = counter["Netral"] if isNeutral == True else 0

    sizes = [positif, netral, negatif]
    labels = ['Positif', 'Netral', 'Negatif']
    colors = ['#00FF00', '#0000FF', '#FF0000']
    plt.pie(sizes, labels=labels, autopct='%1.0f%%',
            shadow=True, colors=colors, textprops={'fontsize': 20})
    plt.savefig('static/files/pie-diagram.png')

    # Bar chart
    plt.figure()
    plt.hist(y, color=('#0000FF'))
    plt.xlabel("Tweet tentang UU KES")
    plt.ylabel("Jumlah Tweet")
    plt.title("Presentase Sentimen Tweet")
    plt.savefig('static/files/bar-diagram.png')
    flash('Klasifikasi Berhasil', 'klasifikasi_data')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preprocessing', methods=['GET', 'POST'])
def preprocessing():
    if request.method == 'POST':
        if request.form.get('upload') == 'Upload Data':
            hasil_preprocessing.clear()
            file = request.files['file']
            if not allowed_file(file.filename):
                flash('Format file tidak diperbolehkan', 'upload_gagal')
                return render_template('preprocessing.html', value=hasil_preprocessing)

            if 'file' not in request.files:
                flash('File tidak boleh kosong', 'upload_gagal')
                return render_template('preprocessing.html', value=hasil_preprocessing)

            if file.filename == '':
                flash('File tidak boleh kosong', 'upload_gagal')
                return render_template('preprocessing.html', value=hasil_preprocessing)

            if file and allowed_file(file.filename):
                file.filename = "Data Scraping.csv"
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                hasil_preprocessing.clear()
                flash('File Berhasil di upload', 'upload_berhasil')
                return render_template('preprocessing.html')

        if request.form.get('preprocess') == 'Preprocessing Data':
            preprocessing_twitter()
            return render_template('preprocessing.html', value=hasil_preprocessing)

    return render_template('preprocessing.html', value=hasil_preprocessing)

@app.route('/labeling', methods=['GET', 'POST'])
def labeling():
    if request.method == 'POST':
        if request.form.get('upload') == 'Upload Data':
            hasil_labeling.clear()
            file = request.files['file']
            if not allowed_file(file.filename):
                flash('Format file tidak diperbolehkan', 'upload_gagal')
                return render_template('labeling.html', value=hasil_labeling)

            if 'file' not in request.files:
                flash('File tidak boleh kosong', 'upload_gagal')
                return render_template('labeling.html', value=hasil_labeling)

            if file.filename == '':
                flash('File tidak boleh kosong', 'upload_gagal')
                return render_template('labeling.html', value=hasil_labeling)

            if file and allowed_file(file.filename):
                file.filename = "Data Preprocessing.csv"
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                hasil_labeling.clear()
                flash('File Berhasil di upload', 'upload_berhasil')
                return render_template('labeling.html')

        if request.form.get('labeling') == 'Labeling Data':
            labeling_twitter()
            return render_template('labeling.html', value=hasil_labeling)

    return render_template('labeling.html', value=hasil_labeling)


@app.route('/klasifikasi', methods=['GET', 'POST'])
def klasifikasi():
    if request.method == 'POST':
        if request.form.get('upload') == 'Upload Data':
            file = request.files['file']
            if not allowed_file(file.filename):
                flash('Format file tidak diperbolehkan', 'upload_gagal')
                return render_template('klasifikasi.html')
            if 'file' not in request.files:
                flash('File tidak boleh kosong', 'upload_gagal')
                return render_template('klasifikasi.html',)
            if file.filename == '':
                flash('File tidak boleh kosong', 'upload_gagal')
                return render_template('klasifikasi.html')
            if file and allowed_file(file.filename):
                file.filename = "Data Labeling.csv"
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                flash('File Berhasil di upload', 'upload_berhasil')
                return render_template('klasifikasi.html')

        if request.form.get('klasifikasi') == 'Klasifikasi Data':
            proses_klasifikasi()
            return render_template('klasifikasi.html', accuracy=akurasi, tables=[df.to_html(classes='table table-bordered', index=False, justify='left')], titles=df.columns.values, tables2=[df2.to_html(classes='table table-bordered', index=False, justify='left')], titles2=df2.columns.values)
            
    if akurasi == 0:
        return render_template('klasifikasi.html')
    else:
        # return render_template('klasifikasi.html', accuracy=akurasi, tables=[df.to_html(classes='table table-bordered', index=False, justify='left')], titles=df.columns.values, tables2=[df2.to_html(classes='table table-bordered', index=False, justify='left')], titles2=df2.columns.values)
        # return render_template('klasifikasi.html', accuracy=akurasi, table1=html_table1, titles1=df.columns.values, table2=html_table2, titles2=df2.columns.values)
        # return render_template('klasifikasi.html', accuracy=akurasi, titles1=df.columns.values, titles2=df2.columns.values)
        return render_template('klasifikasi.html', accuracy=akurasi)



# @app.route('/tfidf_results', methods=['GET', 'POST'])
# def tfidf_results():

    # Get the text data from the klasifikasi function
    texts = [row[1] for row in tweet]  # Assuming tweet contains the classified data

    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer to the texts and transform them into a matrix
    x = vectorizer.fit_transform(texts)

    # Calculate TF-IDF matrix
    tfidf_matrix = x.toarray()

    # Calculate term frequencies
    term_frequencies = np.sum(x.toarray(), axis=0)

    # Calculate document frequencies
    document_frequencies = np.sum(x.toarray() != 0, axis=0)

    # Calculate inverse document frequencies
    inverse_document_frequencies = np.log(x.shape[0] / (1 + document_frequencies))

    # Calculate TF-IDF values
    tf_idf_values = term_frequencies * inverse_document_frequencies

    # Convert TF-IDF matrix to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=vectorizer.get_feature_names_out())

    # Render HTML template with calculated values
    return render_template('tfidf_results.html', 
                           tfidf_data=tfidf_df, 
                           term_frequencies=term_frequencies, 
                           document_frequencies=document_frequencies, 
                           inverse_document_frequencies=inverse_document_frequencies, 
                           tf_idf_values=tf_idf_values)

# @app.route('/tfidf_results',methods=['GET'])
# def tfidf_results():
#     global df
#     global df2
#     global akurasi
#     tweet = []
#     y = []

#     with open("static/files/Data Labeling.csv", encoding='utf-8') as csvfile:
#         readCSV = csv.reader(csvfile, delimiter=',')
#         for row in readCSV:
#             tweet.append(row[1])
#             y.append(row[2])

#     vectorizer = TfidfVectorizer()
#     vectorizer.fit(tweet)
#     # tfidf = vectorizer.fit_transform(X_train)
#     x = vectorizer.transform(tweet)

# # tfidf
# # Calculate term frequencies
#     term_frequencies = np.sum(x.toarray(), axis=0)
# # Calculate document frequencies
#     document_frequencies = np.sum(x.toarray() != 0, axis=0)
# # Calculate inverse document frequencies
#     inverse_document_frequencies = np.log(x.shape[0] / (1 + document_frequencies))
# # Calculate TF-IDF values
#     tf_idf_values = term_frequencies * inverse_document_frequencies
# # Convert TF-IDF matrix to DataFrame
#     tfidf_df = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())
#     return render_template('tfidf_results.html', 
#                            tfidf_data=tfidf_df.to_html(), 
#                            term_frequencies=term_frequencies, 
#                            document_frequencies=document_frequencies, 
#                            inverse_document_frequencies=inverse_document_frequencies, 
#                            tf_idf_values=tf_idf_values)

@app.route('/tfidf_results', methods=['GET'])
def tfidf_results():
    global df_result
    global df
    global df2
    global akurasi
    tweet = []
    y = []

    # Read data from CSV file
    with open("static/files/Data Labeling.csv", encoding='utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            tweet.append(row[1])
            y.append(row[2])

    # Vectorize the text data
    vectorizer = TfidfVectorizer()
    vectorizer.fit(tweet)
    x = vectorizer.transform(tweet)
    
    # Ambil matriks TF-IDF
    tfidf_matrix = x.toarray()

    # Ambil kata-kata yang muncul di korpus
    terms = vectorizer.get_feature_names_out()

    # Ambil DF (Document Frequency) untuk setiap kata
    # # df = vectorizer.idf_
    # print(x.asfptype())
    # print("1===================")
    # print(x.asfptype().tolist()[0])
    # print("2===================")
    # print(len(x.asfptype().tolist()[0]))
    # print("3===================")
    # df = x.asfptype().sum(axis=0).tolist()[0]

    # #  Hitung nilai TF
    # TF = []
    # for i, term in enumerate(terms):
    #     term_count = sum(1 for document in tweet if term in document)
    #     TF.append(term_count)

    # IDF = []
    # for i, term in enumerate(terms):
    #     IDF.append((np.log(df[i]) + 1))

    # TFIDF = []
    # for i, term in enumerate(terms):
    #     TFIDF.append(IDF[i] * TF[i])
    # Hitung nilai DF
    
    df = []
    for i, term in enumerate(terms):
        df_count = sum(1 for document in tweet if term in document)
        df.append(df_count)

    # Hitung nilai TF
    TF = []
    for i, term in enumerate(terms):
        term_freq = []
        for document in tweet:
            term_freq.append(document.count(term))
        TF.append(np.array(term_freq))

    IDF = []
    for i, term in enumerate(terms):
        if df[i] == 0:
            IDF.append(0)
        else:
            # IDF.append((np.log10(len(tweet)/df[i]) + 1))
            IDF.append(np.log(len(tweet) / df[i]) + 1)
                       # IDF.append((np.log10(len(tweet) / (df[i] + 1)) + 1))
            # IDF.append(np.log10(1182/60)+1)


    TFIDF = []
    for i, term in enumerate(terms):
        TFIDF.append(IDF[i] * TF[i])
        #  TFIDF.append(len(tweet))
    # Membuat DataFrame dari hasil perhitungan
    data = {'Kata': terms, 'TF': TF ,'DF': df, 'IDF': IDF, 'TF-IDF' : TFIDF}

    df_result = pd.DataFrame(data)
    
    # Render the template with the DataFrame
    return render_template('tfidf_results.html', tables=[df_result.to_html(classes='table table-bordered', index=False, justify='left')], titles=df_result.columns.values)

@app.route('/visualisasi')
def visualisasi():
    return render_template('visualisasi.html')

@app.route('/tentang')
def modelpredict():
    return render_template('tentang.html')

if __name__ == "__main__":
    app.run(debug=True)