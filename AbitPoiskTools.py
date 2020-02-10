# Standard library imports
from typing import Iterable
from random import sample
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third party imports
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup



# Translates regions' names into English
regions_mapper_v1 = {'АР Крим' : 'KK', 'Волинська область' : 'KC', 'Вінницька область' : 'KB', 'Дніпропетровська область' : 'KE', 'Донецька область' : 'KH', 'Житомирська область' : 'KM', 'Закарпатська область' : 'KO', 'Запорізька область' : 'KP', 'Івано-Франківська область' : 'KT',  'Київська область' : 'KI', 'Кіровоградська область' : 'HA', 'Луганська область' : 'HB',  'Львівська область' : 'HC', 'Миколаївська область' : 'HE', 'Одеська область' : 'HH', 'Полтавська область' : 'HI', 'Рівненська область' : 'HK', 'Сумська область' : 'HM', 'Тернопільська область' : 'HO', 'Харківська область' : 'KX', 'Херсонська область' : 'HT', 'Хмельницька область' : 'HX', 'Черкаська область' : 'IA', 'Чернівецька область' : 'IE', 'Чернігівська область' : 'IB',   'м. Київ' : 'KA',  'м. Севастополь' : 'IH',  'За межами України' : 'Abroad'}

regions_mapper_v2 = {'АР Крим' : 'Crimea', 'Волинська область' : 'Vol.', 'Вінницька область' : 'Vinn.', 'Дніпропетровська область' : 'Dnipr.', 'Донецька область' : 'Don.', 'Житомирська область' : 'Zhyt.', 'Закарпатська область' : 'Zakar.', 'Запорізька область' : 'Zapor.', 'Івано-Франківська область' : 'Ivan.',  'Київська область' : 'Kyiv', 'Кіровоградська область' : 'Kirov.', 'Луганська область' : 'Luhan.',  'Львівська область' : 'Lviv', 'Миколаївська область' : 'Mykol.', 'Одеська область' : 'Odessa', 'Полтавська область' : 'Polt.', 'Рівненська область' : 'Rivne', 'Сумська область' : 'Sumy', 'Тернопільська область' : 'Tern.', 'Харківська область' : 'Khar.', 'Херсонська область' : 'Kher.', 'Хмельницька область' : 'Khmel.', 'Черкаська область' : 'Cherk.', 'Чернівецька область' : 'Cherniv.', 'Чернігівська область' : 'Chernih.',   'м. Київ' : 'Kyiv city',  'м. Севастополь' : 'Sevastopol',  'За межами України' : 'Abroad'}

regions_mapper_v3 = {'АР Крим' : 'Crimea',
                     'Волинська область' : 'Volyn',
                     'Вінницька область' : 'Vinnytsia',
                     'Дніпропетровська область' : 'Dnipropetrovsk',
                     'Донецька область' : 'Donetsk',
                     'Житомирська область' : 'Zhytomyr',
                     'Закарпатська область' : 'Zakarpattia',
                     'Запорізька область' : 'Zaporizhia',
                     'Івано-Франківська область' : 'Ivano-Frankivsk', 
                     'Київська область' : 'Kyiv',
                     'Кіровоградська область' : 'Kirovohrad',
                     'Луганська область' : 'Luhansk', 
                     'Львівська область' : 'Lviv',
                     'Миколаївська область' : 'Mykolaiv',
                     'Одеська область' : 'Odessa',
                     'Полтавська область' : 'Poltava',
                     'Рівненська область' : 'Rivne',
                     'Сумська область' : 'Sumy',
                     'Тернопільська область' : 'Ternopil',
                     'Харківська область' : 'Kharkiv',
                     'Херсонська область' : 'Kherson',
                     'Хмельницька область' : 'Khmelnytskyi',
                     'Черкаська область' : 'Cherkasy',
                     'Чернівецька область' : 'Chernivtsi',
                     'Чернігівська область' : 'Chernihiv',  
                     'м. Київ' : 'Kyiv city', 
                     'м. Севастополь' : 'Sevastopol City', 
                     'За межами України' : 'Abroad'}

subject_mapper = {'Середній бал документа про освіту':'Diploma',
              'Українська мова та література (ЗНО)':'UkrLang',
              'Історія України (ЗНО)':'History',
              'Іноземна мова (ЗНО)':'ForeignLang',
              'Математика (ЗНО)':'Mathematics',
              'Географія (ЗНО)':'Geography',
              'Біологія (ЗНО)':'Biology',
              'Бал за успішне закінчення підготовчих курсів закладу освіти':'Prepcourse',
              'Фізика (ЗНО)':'Physics',
              'Хімія (ЗНО)':'Chemistry',
              'Українська мова та література':'UkrLang',
              'Математика':'Mathematics',
              '--> Середній бал документа про освіту':'Diploma',
              'Іноземна мова':'ForeignLang',
              'Бал за\xa0особливі успіхи':'ExtraCredits',
              'Історія України':'History',
              '--> Математика (ЗНО)':'Mathematics',
              'Географія':'Geography',
              '--> Іноземна мова (ЗНО)':'ForeignLang',
              'Результат ЗНО з української мови та літератури':'UkrLang',
              'Біологія':'Biology',
              'Фізика':'Physics',
              '--> Біологія (ЗНО)':'Biology',
              'Право':'Law',
              '--> Фізика (ЗНО)':'Physics',
              'Англійська мова (ЗНО)':'ForeignLang',
              '--> Історія України (ЗНО)':'History'}

def draw_plotly(G, pos, to_title = '<br>Network graph made with Python'):
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size= 10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))
    
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title=to_title,
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code on the Plotly website can be found <a href='https://plot.ly/ipython-notebooks/network-graphs/'> here</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    fig.show()
    

def get_hrefs(soup, pattern , only_count = False):
    """Extracts links that contain a specified pattern or returns the last page number for page patterns"""
    output = [x.get('href') for x in soup.find_all('a') if isinstance(x.get('href'), str)]
    output = [x for x in output if pattern in x]
    
    if only_count == True:
        if len(output) == 0:
            return None # i.e. no link with a specified patter has been found
        output = list(set(output))
        output = max([int(x.split('?page=')[-1]) for x in output])
    return output

def get_page(link:str, add_key:bool = True, autocomplete:bool = True, extract_pattern:str = None,
             return_soup:bool = True):
    """
    A generic function to scrape a single page from Abit-Poisk. Options:
    add_key - adds a variable part of the input link to "Key" column. If tuple, the elements should be a column name & its value.
    autocomplete -  in case a link is just a variable part, this adds the root URL
    extract_pattern - calls get_hrefs  to extract all links containing a specified pattern
    drop_summary - whether to add an np.nan to links to handle table where the last row is summary statistics with no link
    return_soup - returns a soup together with the dataframe for get_pages function.
    """
    if autocomplete == True:
        link = 'https://abit-poisk.org.ua' + link
    
    response = requests.get(link)
    assert response.status_code == 200, "No repsonse"
    df_lambda = pd.read_html(response.content)[0]
    
    # Drops adds by Google and a summary row
    to_keep = ~df_lambda.iloc[:,0].str.contains('adsbygoogle|Загалом')
    df_lambda = df_lambda.loc[to_keep].reset_index(drop = True)
    
    # Extracts links containing a specified pattern
    if extract_pattern is not None:
        soup = BeautifulSoup(response.content, 'html.parser')
        links = get_hrefs(soup, extract_pattern)
        if df_lambda.shape[0] == len(links)/2: # Ad hoc error handling
            links = links[::2]
        df_lambda['Link'] = links
    
    # Adds a key to link results from regions to universities and from universities to programmes
    if add_key == True:
        df_lambda['Key'] = link.replace('https://abit-poisk.org.ua', '')
    
    # A soup might be needed to check if there are several pages for the table at hand
    if return_soup == True:
        
        # Checking if the soup already exists because we extracted the pattern
        if not 'soup' in locals():
            soup = BeautifulSoup(response.content, 'html.parser')
        return df_lambda, soup
    else:
        return df_lambda
    
def get_multipage(*args, **kwargs):
    """
    A generic function to scrape a regional, university or programme level pages from Abit-Poisk, including multi-paged ones.
    This is a generalisation of get_page and takes same input. return_soup should not be included in kwargs.
    If the scraper encounters a page with no applications (and no table), the exception print the link and returns nothing.
    """
    assert 'return_soup' not in kwargs, 'return_soup should not be used'
    df_lambda, soup  = get_page(*args, **kwargs)
    # If gets the first page, checks if there are more pages to scrape
    last_page = get_hrefs(soup, '?page=', only_count = True)
    if last_page is not None:
        # Loops through each of those, scrapes tables and appends new dataframes
        for x in range(2, last_page+1):
            page = link + f'/?page={x}'
            df_alpha = get_page(page, add_key = ('Key', link), extract_pattern = extract_pattern, return_soup = False)
            df_lambda = df_lambda.append(df_alpha)
    
    return df_lambda
    

def get_multithread(links:Iterable, pattern, nthreads = 4):
    """
    A further generalisation for the scraper. Takes an iterable of links and applies get_multipage to every link.
    Returns a combined pandas dataframe
    """
    lambda_dfs = [] # stores the intermediary results
    
    falpha = partial(get_multipage, extract_pattern = pattern)
    # Creating a pool of workers
    with ThreadPoolExecutor(max_workers = nthreads) as e:
        futures = []
        # Assigning the link to scrape
        for x in links:
            futures.append(e.submit(falpha, x))
            
        # Obtaining the pages as they are scraped
        for x in tqdm(as_completed(futures)):
            lambda_dfs.append(x.result())
        exceptions = [x for x in lambda_dfs if type(x) == str]
        lambda_dfs = [x for x in lambda_dfs if type(x) != str]
        if len(exceptions) > 0:
            k = len(exceptions)
            print(f'{k} exceptions trigerred. Examples:')
            for x in sample(exceptions, k = 10 if k > 10 else k):
                print('https://abit-poisk.org.ua' + x)
        # Creating a single output dataframe
        return pd.concat(lambda_dfs, ignore_index = True, sort = False)
    