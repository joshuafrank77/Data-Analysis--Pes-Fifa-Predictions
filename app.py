from flask import Flask, request, jsonify, render_template,Response,stream_with_context
import pandas as pd
import requests
import joblib
from bs4 import BeautifulSoup
from datetime import datetime
from rembg import remove
from PIL import Image
from io import BytesIO
import base64
from datetime import datetime
from datetime import datetime, timezone
import time
import re
import json
import lightgbm
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
import numpy as np
import cpi
from dateutil.relativedelta import relativedelta




app = Flask(__name__)

# Load the model
model_filename_1 = 'stacking_regressor_model.pkl'
model_filename = 'stacking_regressor_model_model.pkl'
model_sofifa_filename = 'stacking_regressor_model_sofifa.pkl'
loaded_model = joblib.load(model_filename)
loaded_model_sofifa = joblib.load(model_sofifa_filename)

# Load the ranking mapping data
ranking_mapping = pd.read_excel('ranking_mapping.xlsx')
ranking_mapping.rename(columns={
    'Country League': 'country_league',
    'League Name': 'league_name',
    'Continent': 'continent',
    'League_img': 'league_img'  # Add this line
}, inplace=True)



def validate_and_correct_transfermarkt_url(url):
    parsed_url = urlparse(url)
    
    if 'transfermarkt' not in parsed_url.netloc:
        return "Invalid URL: URL must be from transfermarkt"
    
    netloc = 'www.transfermarkt.com'
    path_components = parsed_url.path.strip('/').split('/')
    
    if 'verein' in path_components:
        return "Invalid URL: Please input the correct player URL"
    
    if len(path_components) < 4 or path_components[2] != 'spieler':
        return "Invalid URL: URL structure is incorrect"
    
    player_name = path_components[0]
    player_id = path_components[3]
    
    if not player_id.isdigit():
        return "Invalid URL: Player ID must be a number"
    
    corrected_url = f"https://{netloc}/{player_name}/profil/spieler/{player_id}"
    
    return corrected_url

def remove_background(image_url):
    if image_url == "https://img.a.transfermarkt.technology/portrait/big/default.jpg?lm=1":
        with open('static/default.png', 'rb') as default_img:
            img_data = default_img.read()
            img_str = base64.b64encode(img_data).decode('utf-8')
            return img_str

    response = requests.get(image_url)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    result = remove(image)
    buffered = BytesIO()
    result.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


# Load the clubs data from the JSON URL
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
url = 'https://dataviz.theanalyst.com/opta-power-rankings/pr-reference.json'
response = requests.get(url, headers=HEADERS)
response.raise_for_status()
clubs_data = response.json()

# Define the prediction function
def make_prediction(input_data):
    prediction = loaded_model.predict(input_data)
    return prediction[0]

def make_prediction_sofifa(input_data):
    prediction = loaded_model_sofifa.predict(input_data)
    return prediction[0]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/indexTeam.html')
def index_team():
    return render_template('indexTeam.html')


def market_value_to_number(value_str):
    value_str = value_str.strip().replace('$', '').replace('€', '').replace('K', '000').replace('m', '0000').replace('.', '')
    value_str_lower = value_str.lower()
    if 'm' in value_str_lower:
        return int(float(value_str_lower.replace('m', '')) * 1000000)
    elif 'k' in value_str_lower:
        return int(float(value_str_lower.replace('k', '')) * 1000)
    else:
        return int(value_str_lower)

def convert_url(url):

    parsed_url = urlparse(url)
    
    if 'transfermarkt' not in parsed_url.netloc:
        return "Invalid URL: URL must be from transfermarkt"
    
    netloc = 'www.transfermarkt.com'
    path_components = parsed_url.path.strip('/').split('/')
    
    if 'spieler' in path_components:
        return "Invalid URL: Please input the correct Team URL, it seems like the url is from a player"
    
    if len(path_components) < 4 or path_components[2] != 'verein':
        return "Invalid URL: URL structure is incorrect"
    
    team_name = path_components[0]
    team_id = path_components[3]
    
    if not team_id.isdigit():
        return "Invalid URL: Team ID must be a number, or its missing"
    
    corrected_url = f"https://{netloc}/{team_name}/kader/verein/{team_id}/plus/1"
    
    return corrected_url


def scrape_player_data(player_url):
    response = requests.get(player_url, headers=HEADERS)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extracting CAPS and GOALS
    caps, goals = 0, 0
    for li in soup.find_all('li', class_='data-header__label'):
        # Check for 'Current international:' and youth divisions
        if 'Current international:' in li.text:
            current_int = li.find('a')
            if current_int:
                country_info = current_int.text.strip()
                if any(f'U{n}' in country_info for n in range(15, 24)):  # Check for U15 to U23
                    # If it's a youth division, set CAPS and GOALS to 0 and break the loop
                    caps, goals = 0, 0
                    break

        # Check for 'Caps/Goals:' and extract values
        if 'Caps/Goals:' in li.text:
            caps_goals = li.find_all('a', class_='data-header__content--highlight')
            if caps_goals and len(caps_goals) >= 2:
                caps = int(caps_goals[0].text.strip())
                goals = int(caps_goals[1].text.strip())

    # Extracting profile image
    profile_img = None
    profile_img_div = soup.find('div', class_='data-header__profile-container')
    if profile_img_div:
        img_tag = profile_img_div.find('img')
        if img_tag and 'src' in img_tag.attrs:
            profile_img = img_tag['src'].replace('header', 'big')

    profile_img = remove_background(profile_img)
    if profile_img:
        profile_img = f"data:image/png;base64,{profile_img}"
    else:
        profile_img = None

    country = 'Spain'
    nationality_el = soup.find("span", {"itemprop": "nationality"})
    if nationality_el:
        country = nationality_el.getText().replace("\n", "").strip()

        

    return caps, goals, profile_img ,country

def scrape_team_data(url):
    url = convert_url(url)
    print(f"Scraping team data from URL: {url}")
    response = requests.get(url, headers=HEADERS)
    time.sleep(2)
    player_data = []
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        player_rows = soup.find_all('tr', class_=['odd', 'even'])
        for row in player_rows:
            player_name_tag = row.find('td', class_='hauptlink').find('a')
            player_name = player_name_tag.text.strip() if player_name_tag else ''
            player_id_tag = player_name_tag['href'] if player_name_tag else ''
            player_id = re.search(r'/(\d+)', player_id_tag).group(1) if player_id_tag else ''
            position_tag = row.find_all('td')[1].find_all('tr')[-1].find('td')
            position = position_tag.text.strip() if position_tag else 'Central Midfield'

            if position == 'Defender':
                position = 'Centre-Back'
            if position == 'Striker':
                position = 'Centre-Forward'
            if position == 'Midfielder':
                position = 'Central Midfield'                

            if position in ['Goalkeeper']:
                position_role = 'Goalkeeper'
            elif position in ['Right-Back', 'Left-Back']:
                position_role = 'Full-Back'
            elif position == 'Centre-Back':
                position_role = 'Centre-Back'
            elif position in ['Right Midfield', 'Left Midfield', 'Central Midfield', 'Defensive Midfield', 'Attacking Midfield']:
                position_role = 'Midfield'
            elif position in ['Right Winger', 'Left Winger']:
                position_role = 'Winger'
            elif position in ['Centre-Forward', 'Second Striker']:
                position_role = 'Forward'

            club_name_tag = soup.find('h1', class_='data-header__headline-wrapper')
            club_name = club_name_tag.text.strip() if club_name_tag else ''
            birthday_tag = row.find_all('td', class_='zentriert')[1]
            birthday_text = birthday_tag.text.strip() if birthday_tag else ''
            birthday_match = re.search(r'(\w{3} \d{1,2}, \d{4})', birthday_text)
            if birthday_match:
                birthday = datetime.strptime(birthday_match.group(1), '%b %d, %Y').strftime('%m/%d/%Y')
                age = (datetime.now() - datetime.strptime(birthday_match.group(1), '%b %d, %Y')).days // 365
                age = age if age < 0 else age
            else:
                birthday = ''
                age = 25

            player_url_tag = row.find('td', class_='hauptlink').find('a')
            player_url = f'https://www.transfermarkt.com{player_url_tag["href"]}' if player_url_tag else ''
            market_value_tag = row.find('td', class_='rechts hauptlink').find('a')
            market_value = market_value_to_number(market_value_tag.text.strip()) if market_value_tag else None

            if market_value == '-':
                market_value = 10000

            height_tag = row.find_all('td', class_='zentriert')[3]
            height = height_tag.text.strip().replace('m', '').replace(',', '') if height_tag and height_tag.text.strip() not in ('', '-') else 180

            foot_tag = row.find_all('td', class_='zentriert')[4]
            foot = foot_tag.text.strip() if foot_tag and foot_tag.text.strip()not in ('', '-') else 'right'

            if foot == 'both':
                foot = 'right'

            contract_date_tag = row.find_all('td', class_='zentriert')[7]
            contract_date_text = contract_date_tag.text.strip() if contract_date_tag else ''

            try:
                contract_date = datetime.strptime(contract_date_text, '%b %d, %Y')
            except Exception as e:
                print(f"Error parsing contract date: {e}")
                contract_date = None

            if not contract_date:
                today = datetime.today()
                next_year_jan_1 = datetime(today.year + 1, 1, 1)
                next_year_jul_1 = datetime(today.year + 1, 7, 1)
                days_left_contract = max((next_year_jan_1 - today).days, (next_year_jul_1 - today).days)
            else:
                days_left_contract = (contract_date - datetime.today()).days

            player_data.append({
                'player_id': player_id,
                'Player Name': player_name,
                'Position': position,
                'position_role':position_role,
                'Age': age,
                'MarketValue': market_value,
                'ClubName': club_name,
                'Height': height,
                'Foot': foot,
                'days_left_contract': days_left_contract,
                'Player URL': player_url,
            })
    return player_data



def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

@app.route('/predict_team', methods=['POST'])
def predict_team():
    data = request.form
    club_id = data['club']
    transfermarkt_url = data['transfermarkt_url']
    

    club_details = next((club for club in clubs_data if club['contestantId'] == club_id), None)

    if not club_details:
        return render_template("team_result.html", predictions=[])

    team_rating = club_details['currentRating']
    team_ranking = club_details['rank']
    team_id = club_details['contestantId']
    team_name = club_details['contestantName']

    # Calculate league rating
    league_id = club_details['tmcl']
    league_clubs = [club for club in clubs_data if club['tmcl'] == league_id]
    league_rating = sum(club['currentRating'] for club in league_clubs) / len(league_clubs)

    league_img = ranking_mapping.loc[ranking_mapping['tmcl'] == league_id, 'league_img'].values[0]


    player_data = scrape_team_data(transfermarkt_url)
    team_predictions = []

    country_flags = {
                'Afghanistan': 'https://flagcdn.com/w580/af.png','Albania': 'https://flagcdn.com/w580/al.png','Algeria': 'https://flagcdn.com/w580/dz.png','Andorra': 'https://flagcdn.com/w580/ad.png','Angola': 'https://flagcdn.com/w580/ao.png','Antigua and Barbuda': 'https://flagcdn.com/w580/ag.png','Argentina': 'https://flagcdn.com/w580/ar.png','Armenia': 'https://flagcdn.com/w580/am.png','Aruba': 'https://flagcdn.com/w580/aw.png','Australia': 'https://flagcdn.com/w580/au.png','Austria': 'https://flagcdn.com/w580/at.png','Azerbaijan': 'https://flagcdn.com/w580/az.png',
                'Bangladesh': 'https://flagcdn.com/w580/bd.png','Bahrain':'https://flagpedia.net/data/flags/w580/bh.webp','Barbados': 'https://flagcdn.com/w580/bb.png','Belarus': 'https://flagcdn.com/w580/by.png','Belgium': 'https://flagcdn.com/w580/be.png','Benin': 'https://flagcdn.com/w580/bj.png','Bermuda': 'https://flagcdn.com/w580/bm.png','Bolivia': 'https://flagcdn.com/w580/bo.png',
                'Bosnia-Herzegovina': 'https://flagcdn.com/w580/ba.png', 'Brazil': 'https://flagcdn.com/w580/br.png','Bulgaria': 'https://flagcdn.com/w580/bg.png','Burkina Faso': 'https://flagcdn.com/w580/bf.png','Burundi': 'https://flagcdn.com/w580/bi.png',
                'Cameroon': 'https://flagcdn.com/w580/cm.png','Canada': 'https://flagcdn.com/w580/ca.png','Cape Verde': 'https://flagcdn.com/w580/cv.png','Central African Republic': 'https://flagcdn.com/w580/cf.png','Chad': 'https://flagcdn.com/w580/td.png','Chile': 'https://flagcdn.com/w580/cl.png','China': 'https://flagcdn.com/w580/cn.png','Colombia': 'https://flagcdn.com/w580/co.png','Comoros': 'https://flagcdn.com/w580/km.png','Congo': 'https://flagcdn.com/w580/cg.png','Costa Rica': 'https://flagcdn.com/w580/cr.png','Croatia': 'https://flagcdn.com/w580/hr.png','Cuba': 'https://flagcdn.com/w580/cu.png','Curacao': 'https://flagcdn.com/w580/cw.png','Cyprus': 'https://flagcdn.com/w580/cy.png','Czech Republic': 'https://flagcdn.com/w580/cz.png',
                'Denmark': 'https://flagcdn.com/w580/dk.png','Dominica': 'https://flagcdn.com/w580/dm.png','Dominican Republic': 'https://flagcdn.com/w580/do.png',
                'Ecuador': 'https://flagcdn.com/w580/ec.png','Egypt': 'https://flagcdn.com/w580/eg.png','El Salvador': 'https://flagcdn.com/w580/sv.png','England': 'https://flagcdn.com/w580/gb-eng.png','Equatorial Guinea': 'https://flagcdn.com/w580/gq.png','Eritrea': 'https://flagcdn.com/w580/er.png','Estonia': 'https://flagcdn.com/w580/ee.png','Ethiopia': 'https://flagcdn.com/w580/et.png',
                'Faroe Islands': 'https://flagcdn.com/w580/fo.png','Fiji': 'https://flagcdn.com/w580/fj.png','Finland': 'https://flagcdn.com/w580/fi.png','France': 'https://flagcdn.com/w580/fr.png','French Guiana': 'https://flagcdn.com/w580/gf.png',
                'Gabon': 'https://flagcdn.com/w580/ga.png','Georgia': 'https://flagcdn.com/w580/ge.png','Germany': 'https://flagcdn.com/w580/de.png','Ghana': 'https://flagcdn.com/w580/gh.png','Gibraltar': 'https://flagcdn.com/w580/gi.png','Greece': 'https://flagcdn.com/w580/gr.png','Grenada': 'https://flagcdn.com/w580/gd.png','Guadeloupe': 'https://flagcdn.com/w580/gp.png','Guatemala': 'https://flagcdn.com/w580/gt.png','Guernsey': 'https://flagcdn.com/w580/gg.png','Guinea': 'https://flagcdn.com/w580/gn.png','Guinea-Bissau': 'https://flagcdn.com/w580/gw.png','Guyana': 'https://flagcdn.com/w580/gy.png',
                'Haiti': 'https://flagcdn.com/w580/ht.png','Honduras': 'https://flagcdn.com/w580/hn.png','Hungary': 'https://flagcdn.com/w580/hu.png',
                'Iceland': 'https://flagcdn.com/w580/is.png','India': 'https://flagcdn.com/w580/in.png','Indonesia': 'https://flagcdn.com/w580/id.png','Iran': 'https://flagcdn.com/w580/ir.png','Iraq': 'https://flagcdn.com/w580/iq.png','Ireland': 'https://flagcdn.com/w580/ie.png','Isle of Man': 'https://flagcdn.com/w580/im.png','Israel': 'https://flagcdn.com/w580/il.png','Italy': 'https://flagcdn.com/w580/it.png',
                'Jamaica': 'https://flagcdn.com/w580/jm.png','Japan': 'https://flagcdn.com/w580/jp.png','Jersey': 'https://flagcdn.com/w580/je.png','Jordan': 'https://flagcdn.com/w580/jo.png',
                'Kazakhstan': 'https://flagcdn.com/w580/kz.png','Kenya': 'https://flagcdn.com/w580/ke.png','Kosovo': 'https://flagcdn.com/w580/xk.png','Kyrgyzstan': 'https://flagcdn.com/w580/kg.png',
                'Latvia': 'https://flagcdn.com/w580/lv.png','Lebanon': 'https://flagcdn.com/w580/lb.png','Liberia': 'https://flagcdn.com/w580/lr.png','Libya': 'https://flagcdn.com/w580/ly.png','Lithuania': 'https://flagcdn.com/w580/lt.png','Luxembourg': 'https://flagcdn.com/w580/lu.png',
                'Madagascar': 'https://flagcdn.com/w580/mg.png','Malawi': 'https://flagcdn.com/w580/mw.png','Malaysia': 'https://flagcdn.com/w580/my.png','Mali': 'https://flagcdn.com/w580/ml.png','Malta': 'https://flagcdn.com/w580/mt.png','Martinique': 'https://flagcdn.com/w580/mq.png','Mauritania': 'https://flagcdn.com/w580/mr.png','Mexico': 'https://flagcdn.com/w580/mx.png','Moldova': 'https://flagcdn.com/w580/md.png','Monaco': 'https://flagcdn.com/w580/mc.png','Montenegro': 'https://flagcdn.com/w580/me.png','Montserrat': 'https://flagcdn.com/w580/ms.png','Morocco': 'https://flagcdn.com/w580/ma.png','Mozambique': 'https://flagcdn.com/w580/mz.png',
                'Namibia': 'https://flagcdn.com/w580/na.png','Nepal': 'https://flagcdn.com/w580/np.png','Netherlands': 'https://flagcdn.com/w580/nl.png','New Zealand': 'https://flagcdn.com/w580/nz.png','Nicaragua': 'https://flagcdn.com/w580/ni.png','Nigeria': 'https://flagcdn.com/w580/ng.png','North Macedonia': 'https://flagcdn.com/w580/mk.png','Northern Ireland': 'https://flagcdn.com/w580/gb-nir.png','Norway': 'https://flagcdn.com/w580/no.png',
                'Pakistan': 'https://flagcdn.com/w580/pk.png','Palestine': 'https://flagcdn.com/w580/ps.png','Panama': 'https://flagcdn.com/w580/pa.png','Paraguay': 'https://flagcdn.com/w580/py.png','Peru': 'https://flagcdn.com/w580/pe.png','Philippines': 'https://flagcdn.com/w580/ph.png','Poland': 'https://flagcdn.com/w580/pl.png','Portugal': 'https://flagcdn.com/w580/pt.png','Puerto Rico': 'https://flagcdn.com/w580/pr.png',
                'Réunion': 'https://flagcdn.com/w580/re.png','Romania': 'https://flagcdn.com/w580/ro.png','Russia': 'https://flagcdn.com/w580/ru.png','Rwanda': 'https://flagcdn.com/w580/rw.png',
                'Samoa': 'https://flagcdn.com/w580/ws.png','San Marino':'https://flagpedia.net/data/flags/w580/sm.webp','Sao Tome and Principe': 'https://flagcdn.com/w580/st.png','Saudi Arabia': 'https://flagcdn.com/w580/sa.png','Scotland': 'https://flagcdn.com/w580/gb-sct.png','Senegal': 'https://flagcdn.com/w580/sn.png','Serbia': 'https://flagcdn.com/w580/rs.png','Sierra Leone': 'https://flagcdn.com/w580/sl.png','Singapore': 'https://flagcdn.com/w580/sg.png','Slovakia': 'https://flagcdn.com/w580/sk.png','Slovenia': 'https://flagcdn.com/w580/si.png','Somalia': 'https://flagcdn.com/w580/so.png','South Africa': 'https://flagcdn.com/w580/za.png','Spain': 'https://flagcdn.com/w580/es.png','Sri Lanka': 'https://flagcdn.com/w580/lk.png','St. Kitts & Nevis': 'https://flagcdn.com/w580/kn.png','St. Lucia': 'https://flagcdn.com/w580/lc.png','Sudan': 'https://flagcdn.com/w580/sd.png','Suriname': 'https://flagcdn.com/w580/sr.png','Sweden': 'https://flagcdn.com/w580/se.png','Switzerland': 'https://flagcdn.com/w580/ch.png','Syria': 'https://flagcdn.com/w580/sy.png',
                'Tajikistan': 'https://flagcdn.com/w580/tj.png','Tanzania': 'https://flagcdn.com/w580/tz.png','Thailand': 'https://flagcdn.com/w580/th.png','Togo': 'https://flagcdn.com/w580/tg.png','Trinidad and Tobago': 'https://flagcdn.com/w580/tt.png','Tunisia': 'https://flagcdn.com/w580/tn.png',
                'Uganda': 'https://flagcdn.com/w580/ug.png','Ukraine': 'https://flagcdn.com/w580/ua.png','United Arab Emirates': 'https://flagcdn.com/w580/ae.png','United Kingdom': 'https://flagcdn.com/w580/gb.png','United States': 'https://flagcdn.com/w580/us.png','Uruguay': 'https://flagcdn.com/w580/uy.png','Uzbekistan': 'https://flagcdn.com/w580/uz.png',
                'Vanuatu': 'https://flagcdn.com/w580/vu.png','Venezuela': 'https://flagcdn.com/w580/ve.png','Vietnam': 'https://flagcdn.com/w580/vn.png',
                'Wales': 'https://flagcdn.com/w580/gb-wls.png',
                'Yemen': 'https://flagcdn.com/w580/ye.png',
                'Zambia': 'https://flagcdn.com/w580/zm.png','Zimbabwe': 'https://flagcdn.com/w580/zw.png',
                'Chinese Taipei': 'https://flagcdn.com/w580/tw.png',  # Chinese Taipei (Taiwan)
                'St. Vincent & Grenadinen': 'https://flagcdn.com/w580/vc.png',  # St. Vincent and the Grenadines
                'The Gambia': 'https://flagcdn.com/w580/gm.png',  # Gambia
                'Türkiye': 'https://flagcdn.com/w580/tr.png',  # Turkey
                'Neukaledonien': 'https://flagcdn.com/w580/nc.png',  # New Caledonia
                'Southern Sudan': 'https://flagcdn.com/w580/ss.png',  # South Sudan
                'Hongkong': 'https://flagcdn.com/w580/hk.png',  # Hong Kong
                'Korea, South': 'https://flagcdn.com/w580/kr.png',  # South Korea
                'Timor-Leste': 'https://flagcdn.com/w580/tl.png',  # Timor-Leste
                'DR Congo': 'https://flagcdn.com/w580/cd.png',  # Democratic Republic of the Congo
                'Cote d\'Ivoire': 'https://flagcdn.com/w580/ci.png',  # Ivory Coast
                'Cookinseln': 'https://flagcdn.com/w580/ck.png',  # Cook Islands
                'Bonaire': 'https://flagcdn.com/w580/bq-bo.png',  # Bonaire
                'Bhutan':'https://flagpedia.net/data/flags/w580/bt.webp',
                'Brunei Darussalam':'https://flagpedia.net/data/flags/w580/bn.webp',
                'Cambodia':'https://flagpedia.net/data/flags/w580/kh.webp',
                'Guam':'https://flagpedia.net/data/flags/w580/gu.webp',
                'Kuwait': 'https://flagpedia.net/data/flags/w580/kw.webp',
                'Laos':'https://flagpedia.net/data/flags/w580/la.webp',
                'Macao':'https://flagpedia.net/data/flags/w580/mo.webp',
                'Maldives':'https://flagpedia.net/data/flags/w580/mv.webp',
                'Mongolia':'https://flagpedia.net/data/flags/w580/mn.webp',
                'Myanmar':'https://flagpedia.net/data/flags/w580/mm.webp',
                'Oman':'https://flagpedia.net/data/flags/w580/om.webp',
                'Qatar':'https://flagpedia.net/data/flags/w580/qa.webp',
                'Turkmenistan':'https://flagpedia.net/data/flags/w580/tm.webp',
                'Botsuana':'https://flagpedia.net/data/flags/w580/bw.webp',
                'Djibouti':'https://flagpedia.net/data/flags/w580/dj.webp',
                'Eswatini':'https://flagpedia.net/data/flags/w580/sz.webp',
                'Lesotho':'https://flagpedia.net/data/flags/w580/ls.webp',
                'Mauritius':'https://flagpedia.net/data/flags/w580/mu.webp',
                'Niger':'https://flagpedia.net/data/flags/w580/ne.webp',
                'Belize': 'https://flagpedia.net/data/flags/w580/bz.webp',
                'British Virgin Islands':'https://flagpedia.net/data/flags/w580/vg.webp',
                'Cayman Islands': 'https://flagpedia.net/data/flags/w580/ky.webp',
                'Turks- and Caicosinseln':'https://flagpedia.net/data/flags/w580/tc.webp',
                'Papua New Guinea':'https://flagpedia.net/data/flags/w580/pg.webp',
                'Solomon Islands':'https://flagpedia.net/data/flags/w580/sb.webp',
                'Liechtenstein' :'https://flagpedia.net/data/flags/w580/li.webp',
                'Tahiti': 'https://flagpedia.net/data/flags/w580/pf.webp'
    }

    for player in player_data:
        player_id = player['player_id']
        player_url = player['Player URL']
        caps, goals, profile_img, country = scrape_player_data(player_url)

        flag_url = country_flags.get(country, "")

        api_url = f"https://www.transfermarkt.com/ceapi/marketValueDevelopment/graph/{player_id}"
        response = requests.get(api_url, headers=HEADERS)
        data = response.json()

        if response.status_code != 200:
            continue

        fallback_value = 10000
        if not data.get('list') or (data.get('current') == '-' and data.get('highest') == '-'):
            market_value = fallback_value
            scraped_data = {
                'height': player['Height'],
                'age': player['Age'],
                'position': player['Position'],
                'position_role': player['position_role'],
                'foot': player['Foot'],
                'caps': caps,
                'goals': goals,
                'MarketValue': market_value,
                'days_left_contract': player['days_left_contract'],
                'highest_market_value': market_value,
                'age_at_highest_value': player['Age'],
                'number_of_changes': 1,
                'latest_value': market_value,
                'mean_value': market_value,
                'median_value': market_value,
                'std_deviation': 0,
                'total_increase': 0,
                'total_decrease': 0,
                'current_to_max_ratio': 1,
                'duration_at_max_value': 0,
                'trend': 'stable',
                'profile_img': profile_img if profile_img else "",
                'player_name': player['Player Name'],
                'flag_url': flag_url,
                'team_id': team_id,
                'team_name': team_name,
                'Nationality':country
            }
        else:
            today = datetime.today()
            cutoff_date = today.timestamp() * 1000
            filtered_data = [entry for entry in data['list'] if entry['x'] <= cutoff_date]

            if not filtered_data:
                market_value = fallback_value
                scraped_data = {
                    'height': player['Height'],
                    'age': player['Age'],
                    'position': player['Position'],
                    'position_role': player['position_role'],
                    'foot': player['Foot'],
                    'caps': caps,
                    'goals': goals,
                    'MarketValue': market_value,
                    'days_left_contract': player['days_left_contract'],
                    'highest_market_value': market_value,
                    'age_at_highest_value': player['Age'],
                    'number_of_changes': 1,
                    'latest_value': market_value,
                    'mean_value': market_value,
                    'median_value': market_value,
                    'std_deviation': 0,
                    'total_increase': 0,
                    'total_decrease': 0,
                    'current_to_max_ratio': 1,
                    'duration_at_max_value': 0,
                    'trend': 'stable',
                    'profile_img': profile_img if profile_img else "",
                    'player_name': player['Player Name'],
                    'flag_url': flag_url,
                    'team_id': team_id,
                    'team_name': team_name,
                    'Nationality':country
                }
            else:
                for i in range(1, len(filtered_data)):
                    if filtered_data[i]['y'] == 0:
                        filtered_data[i]['y'] = filtered_data[i-1]['y']

                highest_value = max(entry['y'] for entry in filtered_data)
                age_at_highest_value = next(entry['age'] for entry in filtered_data if entry['y'] == highest_value)
                number_of_changes = len(filtered_data)
                latest_value = filtered_data[-1]['y']

                values = [entry['y'] for entry in filtered_data]
                mean_value = sum(values) / len(values)
                median_value = sorted(values)[len(values) // 2]
                std_deviation = pd.Series(values).std()

                initial_value = 0
                total_increase = sum(max(0, values[i] - (values[i - 1] if i > 0 else initial_value)) for i in range(len(values)))
                total_decrease = sum(max(0, (values[i - 1] if i > 0 else initial_value) - values[i]) for i in range(len(values)))

                current_to_max_ratio = latest_value / highest_value if highest_value != 0 else 0

                max_value_dates = [entry['x'] for entry in filtered_data if entry['y'] == highest_value]
                start_date_of_max_value = min(max_value_dates) if max_value_dates else None
                end_date_of_max_value = None

                if latest_value == highest_value:
                    end_date_of_max_value = cutoff_date
                else:
                    indices_of_max_value = [i for i, entry in enumerate(filtered_data) if entry['y'] == highest_value]
                    max_value_last_index = max(indices_of_max_value) if indices_of_max_value else -1

                    if max_value_last_index < len(filtered_data) - 1:
                        end_date_of_max_value = filtered_data[max_value_last_index + 1]['x']
                    else:
                        end_date_of_max_value = cutoff_date

                if start_date_of_max_value and end_date_of_max_value:
                    duration_at_max_value = (end_date_of_max_value - start_date_of_max_value) / (1000 * 60 * 60 * 24)
                else:
                    duration_at_max_value = 0

                recent_values = values[-5:]
                changes = [recent_values[i] - recent_values[i - 1] for i in range(1, len(recent_values))]

                if highest_value > 50000000:
                    significant_change_threshold = 0.02
                elif highest_value > 5000000:
                    significant_change_threshold = 0.05
                else:
                    significant_change_threshold = 0.10

                average_change = sum(changes) / len(changes) if changes else 0
                if abs(average_change) / latest_value <= significant_change_threshold:
                    trend = "stable"
                elif average_change > 0:
                    trend = "increasing"
                else:
                    trend = "decreasing"

                scraped_data = {
                    'height': player['Height'],
                    'age': player['Age'],
                    'position': player['Position'],
                    'position_role': player['position_role'],
                    'foot': player['Foot'],
                    'caps': caps,
                    'goals': goals,
                    'MarketValue': latest_value,
                    'days_left_contract': player['days_left_contract'],
                    'highest_market_value': highest_value,
                    'age_at_highest_value': age_at_highest_value,
                    'number_of_changes': number_of_changes,
                    'latest_value': latest_value,
                    'mean_value': mean_value,
                    'median_value': median_value,
                    'std_deviation': std_deviation,
                    'total_increase': total_increase,
                    'total_decrease': total_decrease,
                    'current_to_max_ratio': current_to_max_ratio,
                    'duration_at_max_value': duration_at_max_value,
                    'trend': trend,
                    'profile_img': profile_img if profile_img else "",
                    'player_name': player['Player Name'],
                    'flag_url': flag_url,
                    'team_id': team_id,
                    'team_name': team_name,
                    'Nationality': country
                }

        team_predictions.append(scraped_data)

    predictions = []

    for player_data in team_predictions:
        input_data = pd.DataFrame({
            'Height': [int(player_data['height'])],
            'Country': [country],
            'CAPS': [int(player_data['caps'])],
            'CAPS GOALS': [int(player_data['goals'])],
            'Foot': [player_data['foot']],
            'Age': [int(player_data['age'])],
            'Position': [player_data['position']],
            'Position Role': [player_data['position_role']],
            'MarketValue': [float(player_data['MarketValue'])],
            'DaysLeftofContract': [float(player_data['days_left_contract'])],
            'TEAM RATING': [float(team_rating)],
            'TEAM RANKING': [int(team_ranking)],
            'LEAGUE RATING': [float(league_rating)],
            'HighestMarketValue': [float(player_data['highest_market_value'])],
            'AgeAtHighestMarketValue': [int(player_data['age_at_highest_value'])],
            'NumberOfMarketValueChanges': [int(player_data['number_of_changes'])],
            'LatestMarketValue': [float(player_data['latest_value'])],
            'MeanMarketValue': [float(player_data['mean_value'])],
            'MedianMarketValue': [float(player_data['median_value'])],
            'MarketValueStdDeviation': [float(player_data['std_deviation'])],
            'TotalIncrease': [float(player_data['total_increase'])],
            'TotalDecrease': [float(player_data['total_decrease'])],
            'CurrentToMaxRatio': [float(player_data['current_to_max_ratio'])],
            'DurationAtMaxValue': [float(player_data['duration_at_max_value'])],
            'MarketValueTrend': [player_data['trend']]
        })

        try:
            prediction = make_prediction(input_data)  # Implement your prediction function
            prediction_sofifa= make_prediction_sofifa(input_data)
        except Exception as e:
            prediction = f"Error: {str(e)}"

        player_data['prediction'] = prediction
        player_data['prediction_sofifa'] = prediction_sofifa
        predictions.append(player_data)


    return render_template("team_result.html", predictions=predictions,league_img=league_img)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    transfermarkt_url = data['transfermarkt_url']
    
    transfermarkt_url = validate_and_correct_transfermarkt_url(transfermarkt_url)
    if "Invalid URL" in transfermarkt_url:
        return jsonify({"error": transfermarkt_url}), 400
    
    scraped_data = scrape_transfermarkt(transfermarkt_url)
    if not scraped_data:
        return render_template('result.html', prediction="Error scraping data from Transfermarkt", form_data=data, scraped_data={})

    try:
        # Extract club details from the form data
        club_id = data['club']
        club_details = next((club for club in clubs_data if club['contestantId'] == club_id), None)
        
        if not club_details:
            raise ValueError(f"Club data not found for: {club_id}")

        team_rating = club_details['currentRating']
        team_ranking = club_details['rank']
        team_id = club_details['contestantId']
        team_name = club_details['contestantName']
        
        # Calculate league rating
        league_id = club_details['tmcl']

        league_clubs = [club for club in clubs_data if club['tmcl'] == league_id]
        league_rating = sum(club['currentRating'] for club in league_clubs) / len(league_clubs)

        league_img = ranking_mapping.loc[ranking_mapping['tmcl'] == league_id, 'league_img'].values[0]


        # Get country flag URL from index.html
        country = scraped_data['nationality']
        country_flags = {
                'Afghanistan': 'https://flagcdn.com/w580/af.png','Albania': 'https://flagcdn.com/w580/al.png','Algeria': 'https://flagcdn.com/w580/dz.png','Andorra': 'https://flagcdn.com/w580/ad.png','Angola': 'https://flagcdn.com/w580/ao.png','Antigua and Barbuda': 'https://flagcdn.com/w580/ag.png','Argentina': 'https://flagcdn.com/w580/ar.png','Armenia': 'https://flagcdn.com/w580/am.png','Aruba': 'https://flagcdn.com/w580/aw.png','Australia': 'https://flagcdn.com/w580/au.png','Austria': 'https://flagcdn.com/w580/at.png','Azerbaijan': 'https://flagcdn.com/w580/az.png',
                'Bangladesh': 'https://flagcdn.com/w580/bd.png','Bahrain':'https://flagpedia.net/data/flags/w580/bh.webp','Barbados': 'https://flagcdn.com/w580/bb.png','Belarus': 'https://flagcdn.com/w580/by.png','Belgium': 'https://flagcdn.com/w580/be.png','Benin': 'https://flagcdn.com/w580/bj.png','Bermuda': 'https://flagcdn.com/w580/bm.png','Bolivia': 'https://flagcdn.com/w580/bo.png',
                'Bosnia-Herzegovina': 'https://flagcdn.com/w580/ba.png', 'Brazil': 'https://flagcdn.com/w580/br.png','Bulgaria': 'https://flagcdn.com/w580/bg.png','Burkina Faso': 'https://flagcdn.com/w580/bf.png','Burundi': 'https://flagcdn.com/w580/bi.png',
                'Cameroon': 'https://flagcdn.com/w580/cm.png','Canada': 'https://flagcdn.com/w580/ca.png','Cape Verde': 'https://flagcdn.com/w580/cv.png','Central African Republic': 'https://flagcdn.com/w580/cf.png','Chad': 'https://flagcdn.com/w580/td.png','Chile': 'https://flagcdn.com/w580/cl.png','China': 'https://flagcdn.com/w580/cn.png','Colombia': 'https://flagcdn.com/w580/co.png','Comoros': 'https://flagcdn.com/w580/km.png','Congo': 'https://flagcdn.com/w580/cg.png','Costa Rica': 'https://flagcdn.com/w580/cr.png','Croatia': 'https://flagcdn.com/w580/hr.png','Cuba': 'https://flagcdn.com/w580/cu.png','Curacao': 'https://flagcdn.com/w580/cw.png','Cyprus': 'https://flagcdn.com/w580/cy.png','Czech Republic': 'https://flagcdn.com/w580/cz.png',
                'Denmark': 'https://flagcdn.com/w580/dk.png','Dominica': 'https://flagcdn.com/w580/dm.png','Dominican Republic': 'https://flagcdn.com/w580/do.png',
                'Ecuador': 'https://flagcdn.com/w580/ec.png','Egypt': 'https://flagcdn.com/w580/eg.png','El Salvador': 'https://flagcdn.com/w580/sv.png','England': 'https://flagcdn.com/w580/gb-eng.png','Equatorial Guinea': 'https://flagcdn.com/w580/gq.png','Eritrea': 'https://flagcdn.com/w580/er.png','Estonia': 'https://flagcdn.com/w580/ee.png','Ethiopia': 'https://flagcdn.com/w580/et.png',
                'Faroe Islands': 'https://flagcdn.com/w580/fo.png','Fiji': 'https://flagcdn.com/w580/fj.png','Finland': 'https://flagcdn.com/w580/fi.png','France': 'https://flagcdn.com/w580/fr.png','French Guiana': 'https://flagcdn.com/w580/gf.png',
                'Gabon': 'https://flagcdn.com/w580/ga.png','Georgia': 'https://flagcdn.com/w580/ge.png','Germany': 'https://flagcdn.com/w580/de.png','Ghana': 'https://flagcdn.com/w580/gh.png','Gibraltar': 'https://flagcdn.com/w580/gi.png','Greece': 'https://flagcdn.com/w580/gr.png','Grenada': 'https://flagcdn.com/w580/gd.png','Guadeloupe': 'https://flagcdn.com/w580/gp.png','Guatemala': 'https://flagcdn.com/w580/gt.png','Guernsey': 'https://flagcdn.com/w580/gg.png','Guinea': 'https://flagcdn.com/w580/gn.png','Guinea-Bissau': 'https://flagcdn.com/w580/gw.png','Guyana': 'https://flagcdn.com/w580/gy.png',
                'Haiti': 'https://flagcdn.com/w580/ht.png','Honduras': 'https://flagcdn.com/w580/hn.png','Hungary': 'https://flagcdn.com/w580/hu.png',
                'Iceland': 'https://flagcdn.com/w580/is.png','India': 'https://flagcdn.com/w580/in.png','Indonesia': 'https://flagcdn.com/w580/id.png','Iran': 'https://flagcdn.com/w580/ir.png','Iraq': 'https://flagcdn.com/w580/iq.png','Ireland': 'https://flagcdn.com/w580/ie.png','Isle of Man': 'https://flagcdn.com/w580/im.png','Israel': 'https://flagcdn.com/w580/il.png','Italy': 'https://flagcdn.com/w580/it.png',
                'Jamaica': 'https://flagcdn.com/w580/jm.png','Japan': 'https://flagcdn.com/w580/jp.png','Jersey': 'https://flagcdn.com/w580/je.png','Jordan': 'https://flagcdn.com/w580/jo.png',
                'Kazakhstan': 'https://flagcdn.com/w580/kz.png','Kenya': 'https://flagcdn.com/w580/ke.png','Kosovo': 'https://flagcdn.com/w580/xk.png','Kyrgyzstan': 'https://flagcdn.com/w580/kg.png',
                'Latvia': 'https://flagcdn.com/w580/lv.png','Lebanon': 'https://flagcdn.com/w580/lb.png','Liberia': 'https://flagcdn.com/w580/lr.png','Libya': 'https://flagcdn.com/w580/ly.png','Lithuania': 'https://flagcdn.com/w580/lt.png','Luxembourg': 'https://flagcdn.com/w580/lu.png',
                'Madagascar': 'https://flagcdn.com/w580/mg.png','Malawi': 'https://flagcdn.com/w580/mw.png','Malaysia': 'https://flagcdn.com/w580/my.png','Mali': 'https://flagcdn.com/w580/ml.png','Malta': 'https://flagcdn.com/w580/mt.png','Martinique': 'https://flagcdn.com/w580/mq.png','Mauritania': 'https://flagcdn.com/w580/mr.png','Mexico': 'https://flagcdn.com/w580/mx.png','Moldova': 'https://flagcdn.com/w580/md.png','Monaco': 'https://flagcdn.com/w580/mc.png','Montenegro': 'https://flagcdn.com/w580/me.png','Montserrat': 'https://flagcdn.com/w580/ms.png','Morocco': 'https://flagcdn.com/w580/ma.png','Mozambique': 'https://flagcdn.com/w580/mz.png',
                'Namibia': 'https://flagcdn.com/w580/na.png','Nepal': 'https://flagcdn.com/w580/np.png','Netherlands': 'https://flagcdn.com/w580/nl.png','New Zealand': 'https://flagcdn.com/w580/nz.png','Nicaragua': 'https://flagcdn.com/w580/ni.png','Nigeria': 'https://flagcdn.com/w580/ng.png','North Macedonia': 'https://flagcdn.com/w580/mk.png','Northern Ireland': 'https://flagcdn.com/w580/gb-nir.png','Norway': 'https://flagcdn.com/w580/no.png',
                'Pakistan': 'https://flagcdn.com/w580/pk.png','Palestine': 'https://flagcdn.com/w580/ps.png','Panama': 'https://flagcdn.com/w580/pa.png','Paraguay': 'https://flagcdn.com/w580/py.png','Peru': 'https://flagcdn.com/w580/pe.png','Philippines': 'https://flagcdn.com/w580/ph.png','Poland': 'https://flagcdn.com/w580/pl.png','Portugal': 'https://flagcdn.com/w580/pt.png','Puerto Rico': 'https://flagcdn.com/w580/pr.png',
                'Réunion': 'https://flagcdn.com/w580/re.png','Romania': 'https://flagcdn.com/w580/ro.png','Russia': 'https://flagcdn.com/w580/ru.png','Rwanda': 'https://flagcdn.com/w580/rw.png',
                'Samoa': 'https://flagcdn.com/w580/ws.png','San Marino':'https://flagpedia.net/data/flags/w580/sm.webp','Sao Tome and Principe': 'https://flagcdn.com/w580/st.png','Saudi Arabia': 'https://flagcdn.com/w580/sa.png','Scotland': 'https://flagcdn.com/w580/gb-sct.png','Senegal': 'https://flagcdn.com/w580/sn.png','Serbia': 'https://flagcdn.com/w580/rs.png','Sierra Leone': 'https://flagcdn.com/w580/sl.png','Singapore': 'https://flagcdn.com/w580/sg.png','Slovakia': 'https://flagcdn.com/w580/sk.png','Slovenia': 'https://flagcdn.com/w580/si.png','Somalia': 'https://flagcdn.com/w580/so.png','South Africa': 'https://flagcdn.com/w580/za.png','Spain': 'https://flagcdn.com/w580/es.png','Sri Lanka': 'https://flagcdn.com/w580/lk.png','St. Kitts & Nevis': 'https://flagcdn.com/w580/kn.png','St. Lucia': 'https://flagcdn.com/w580/lc.png','Sudan': 'https://flagcdn.com/w580/sd.png','Suriname': 'https://flagcdn.com/w580/sr.png','Sweden': 'https://flagcdn.com/w580/se.png','Switzerland': 'https://flagcdn.com/w580/ch.png','Syria': 'https://flagcdn.com/w580/sy.png',
                'Tajikistan': 'https://flagcdn.com/w580/tj.png','Tanzania': 'https://flagcdn.com/w580/tz.png','Thailand': 'https://flagcdn.com/w580/th.png','Togo': 'https://flagcdn.com/w580/tg.png','Trinidad and Tobago': 'https://flagcdn.com/w580/tt.png','Tunisia': 'https://flagcdn.com/w580/tn.png',
                'Uganda': 'https://flagcdn.com/w580/ug.png','Ukraine': 'https://flagcdn.com/w580/ua.png','United Arab Emirates': 'https://flagcdn.com/w580/ae.png','United Kingdom': 'https://flagcdn.com/w580/gb.png','United States': 'https://flagcdn.com/w580/us.png','Uruguay': 'https://flagcdn.com/w580/uy.png','Uzbekistan': 'https://flagcdn.com/w580/uz.png',
                'Vanuatu': 'https://flagcdn.com/w580/vu.png','Venezuela': 'https://flagcdn.com/w580/ve.png','Vietnam': 'https://flagcdn.com/w580/vn.png',
                'Wales': 'https://flagcdn.com/w580/gb-wls.png',
                'Yemen': 'https://flagcdn.com/w580/ye.png',
                'Zambia': 'https://flagcdn.com/w580/zm.png','Zimbabwe': 'https://flagcdn.com/w580/zw.png',
                'Chinese Taipei': 'https://flagcdn.com/w580/tw.png',  # Chinese Taipei (Taiwan)
                'St. Vincent & Grenadinen': 'https://flagcdn.com/w580/vc.png',  # St. Vincent and the Grenadines
                'The Gambia': 'https://flagcdn.com/w580/gm.png',  # Gambia
                'Türkiye': 'https://flagcdn.com/w580/tr.png',  # Turkey
                'Neukaledonien': 'https://flagcdn.com/w580/nc.png',  # New Caledonia
                'Southern Sudan': 'https://flagcdn.com/w580/ss.png',  # South Sudan
                'Hongkong': 'https://flagcdn.com/w580/hk.png',  # Hong Kong
                'Korea, South': 'https://flagcdn.com/w580/kr.png',  # South Korea
                'Timor-Leste': 'https://flagcdn.com/w580/tl.png',  # Timor-Leste
                'DR Congo': 'https://flagcdn.com/w580/cd.png',  # Democratic Republic of the Congo
                'Cote d\'Ivoire': 'https://flagcdn.com/w580/ci.png',  # Ivory Coast
                'Cookinseln': 'https://flagcdn.com/w580/ck.png',  # Cook Islands
                'Bonaire': 'https://flagcdn.com/w580/bq-bo.png',  # Bonaire
                'Bhutan':'https://flagpedia.net/data/flags/w580/bt.webp',
                'Brunei Darussalam':'https://flagpedia.net/data/flags/w580/bn.webp',
                'Cambodia':'https://flagpedia.net/data/flags/w580/kh.webp',
                'Guam':'https://flagpedia.net/data/flags/w580/gu.webp',
                'Kuwait': 'https://flagpedia.net/data/flags/w580/kw.webp',
                'Laos':'https://flagpedia.net/data/flags/w580/la.webp',
                'Macao':'https://flagpedia.net/data/flags/w580/mo.webp',
                'Maldives':'https://flagpedia.net/data/flags/w580/mv.webp',
                'Mongolia':'https://flagpedia.net/data/flags/w580/mn.webp',
                'Myanmar':'https://flagpedia.net/data/flags/w580/mm.webp',
                'Oman':'https://flagpedia.net/data/flags/w580/om.webp',
                'Qatar':'https://flagpedia.net/data/flags/w580/qa.webp',
                'Turkmenistan':'https://flagpedia.net/data/flags/w580/tm.webp',
                'Botsuana':'https://flagpedia.net/data/flags/w580/bw.webp',
                'Djibouti':'https://flagpedia.net/data/flags/w580/dj.webp',
                'Eswatini':'https://flagpedia.net/data/flags/w580/sz.webp',
                'Lesotho':'https://flagpedia.net/data/flags/w580/ls.webp',
                'Mauritius':'https://flagpedia.net/data/flags/w580/mu.webp',
                'Niger':'https://flagpedia.net/data/flags/w580/ne.webp',
                'Belize': 'https://flagpedia.net/data/flags/w580/bz.webp',
                'British Virgin Islands':'https://flagpedia.net/data/flags/w580/vg.webp',
                'Cayman Islands': 'https://flagpedia.net/data/flags/w580/ky.webp',
                'Turks- and Caicosinseln':'https://flagpedia.net/data/flags/w580/tc.webp',
                'Papua New Guinea':'https://flagpedia.net/data/flags/w580/pg.webp',
                'Solomon Islands':'https://flagpedia.net/data/flags/w580/sb.webp',
                'Liechtenstein' :'https://flagpedia.net/data/flags/w580/li.webp',
                'Tahiti': 'https://flagpedia.net/data/flags/w580/pf.webp'
        }

        flag_url = country_flags.get(country, "")

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}", form_data=data, scraped_data=scraped_data)

    input_data = pd.DataFrame({
        'Height': [int(scraped_data['height'])],
        'Country': [country],
        'CAPS': [int(scraped_data['caps'])],
        'CAPS GOALS': [int(scraped_data['goals'])],
        'Foot': [scraped_data['foot']],
        'Age': [int(scraped_data['age'])],
        'Position': [scraped_data['position']],
        'Position Role': [scraped_data['position_role']],
        'MarketValue': [float(scraped_data['MarketValue'])],
        'DaysLeftofContract': [float(scraped_data['days_left_contract'])],
        'TEAM RATING': [float(team_rating)],
        'TEAM RANKING': [int(team_ranking)],
        'LEAGUE RATING': [float(league_rating)],
        'HighestMarketValue': [float(scraped_data['highest_market_value'])],
        'AgeAtHighestMarketValue': [int(scraped_data['age_at_highest_value'])],
        'NumberOfMarketValueChanges': [int(scraped_data['number_of_changes'])],
        'LatestMarketValue': [float(scraped_data['latest_value'])],
        'MeanMarketValue': [float(scraped_data['mean_value'])],
        'MedianMarketValue': [float(scraped_data['median_value'])],
        'MarketValueStdDeviation': [float(scraped_data['std_deviation'])],
        'TotalIncrease': [float(scraped_data['total_increase'])],
        'TotalDecrease': [float(scraped_data['total_decrease'])],
        'CurrentToMaxRatio': [float(scraped_data['current_to_max_ratio'])],
        'DurationAtMaxValue': [float(scraped_data['duration_at_max_value'])],
        'MarketValueTrend': [scraped_data['trend']]
    })

    try:
        prediction = make_prediction(input_data)
        prediction_sofifa= make_prediction_sofifa(input_data)
    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}", form_data=data, scraped_data=scraped_data ,prediction_sofifa=prediction_sofifa)

    return render_template('result.html', prediction=prediction, form_data=data, scraped_data=scraped_data, 
                               team_rating=team_rating, team_ranking=team_ranking, league_rating=league_rating,
                               team_id=team_id, team_name=team_name, flag_url=flag_url, country = country,league_img=league_img,prediction_sofifa=prediction_sofifa)




inflation_rates = {
    # 1980: 111.421637,
    # 1981: 101.707734,
    # 1982: 92.587577,
    # 1983: 84.042742,
    # 1984: 76.054773,
    # 1985: 68.605188,
    # 1986: 61.675475,
    # 1987: 55.247094,
    # 1988: 49.301475,
    # 1989: 43.820020,
    # 1990: 38.784101,
    # 1991: 34.175064,
    # 1992: 29.974223,


    1993: 31.43271169,
    1994: 26.91421391,
    1995: 16.38618162,
    1996: 9.678047926,
    1997: 10.83845327,
    1998: 13.69234374,
    1999: 8.885190649,
    2000: 8.715823783,
    2001: 6.468291437,
    2002: 5.171477184,
    2003: 5.781808687,
    2004: 5.877685249,
    2005: 6.015680875,
    2006: 5.558162201,
    2007: 6.041907677,
    2008: 4.371529849,
    2009: 2.970615067,
    2010: 3.335709635,
    2011: 2.61844711,
    2012: 3.437112291,
    2013: 3.106689736,
    2014: 1.941098198,
    2015: 1.654583029,
    2016: 1.852637901,
    2017: 0.767866061,
    2018: 0.231478199,
    2019: 0.295112404,
    2020: 0.185157085,
    2021: 0.188444426,
    2022: 0.183684949,
    2023: 0,
    2024: 0,

    # 1993: 31.43271169,
    # 1994: 26.91421391,
    # 1995: 16.38618162,
    # 1996: 13.69234374,
    # 1997: 10.83845327,
    # 1998: 9.678047926,
    # 1999: 8.885190649,
    # 2000: 8.715823783,
    # 2001: 6.468291437,
    # 2002: 6.041907677,
    # 2003: 6.015680875,
    # 2004: 5.877685249,
    # 2005: 5.781808687,
    # 2006: 5.558162201,
    # 2007: 5.171477184,
    # 2008: 4.371529849,
    # 2009: 3.437112291,
    # 2010: 3.335709635,
    # 2011: 3.106689736,
    # 2012: 2.970615067,
    # 2013: 2.61844711,
    # 2014: 1.941098198,
    # 2015: 1.852637901,
    # 2016: 1.654583029,
    # 2017: 0.767866061,
    # 2018: 0.295112404,
    # 2019: 0.231478199,
    # 2020: 0.188444426,
    # 2021: 0.185157085,
    # 2022: 0.183684949,
    # 2023: 0,
    # 2024: 0
}

inflation_df = pd.DataFrame(list(inflation_rates.items()), columns=['Year', 'InflationRate'])

def adjust_for_inflation(value, start_year, end_year=2024):
# def adjust_for_inflation(value,year):
    # inflation_rate = inflation_df[inflation_df['Year'] == year]['InflationRate'].values[0]
    # value = (value * (inflation_rate)) + value
    # return value
    for year in range(start_year, end_year + 1):
        if year in inflation_rates:
            inflation_rate = inflation_rates[year]
            value *= (1 + inflation_rate /100)
    return value




@app.route('/player-at-prime', methods=['POST'])
def player_at_prime():
    data = request.form.to_dict()
    scraped_data = {key.replace('scraped_', ''): value for key, value in data.items() if key.startswith('scraped_')}
    league_img = data.get('league_img')

    try:
        highest_market_value = float(scraped_data['highest_market_value'])
        age_at_highest_value = int(scraped_data['age_at_highest_value'])
        duration_at_max_value = float(scraped_data['duration_at_max_value'])

        scraped_data.update({
            'MarketValue': highest_market_value,
            'LatestMarketValue': highest_market_value,
            'Age': age_at_highest_value,
            'CurrentToMaxRatio': 1.0,
            'duration_at_max_value':duration_at_max_value
        })

        player_id = data['transfermarkt_url'].split('/')[-1]
        api_url_market_value = f"https://www.transfermarkt.com/ceapi/marketValueDevelopment/graph/{player_id}"
        response_market_value = requests.get(api_url_market_value, headers=HEADERS)
        data_market_value = response_market_value.json()
        if response_market_value.status_code != 200:
            return 'FAILED'

        market_values = data_market_value['list']

        # Determine birth year from market values
        first_entry = market_values[0]
        first_entry_date = datetime.utcfromtimestamp(first_entry['x'] / 1000)
        birth_year = first_entry_date.year - int(first_entry['age'])

        # Fetch additional transfer history data
        api_url_transfers = f"https://www.transfermarkt.com/ceapi/transferHistory/list/{player_id}"
        response_transfers = requests.get(api_url_transfers, headers=HEADERS)
        transfer_data = response_transfers.json()
        if response_transfers.status_code != 200:
            return 'FAILED'
        
        # Extract and transform valid transfers
        transfers = transfer_data['transfers']
        for transfer in transfers:
            transfer_date = datetime.strptime(transfer['dateUnformatted'], '%Y-%m-%d')
            if transfer_date < datetime(2004, 10, 4):
                try:
                    fee_value = market_value_to_number(transfer['fee'])
                    age_at_transfer = transfer_date.year - birth_year
                    market_values.append({
                        'x': int(transfer_date.timestamp() * 1000),
                        'y': fee_value,
                        'mw': transfer['fee'],
                        'datum_mw': transfer['date'],
                        'verein': transfer['to']['clubName'],
                        'age': age_at_transfer,
                        'wappen': transfer['to']['clubEmblem-1x']
                    })
                except ValueError:
                    continue

        # Sort market values by timestamp
        market_values.sort(key=lambda x: x['x'])

         #Adjust market values using custom inflation rates
        for mv in market_values:
            market_date = datetime.fromtimestamp(mv['x'] / 1000, tz=timezone.utc).date()
            market_year = market_date.year
            mv['y'] = adjust_for_inflation(mv['y'], market_year)
            mv['age'] = int(mv['age'])

            try:
                # Adjust the market value using the cpi library

                mv['y'] = cpi.inflate(mv['y'], market_year)
                
            except Exception as e:
                # If any error occurs, leave the value unchanged and print the error
                print(f"Error adjusting for inflation for the year {market_year}: {e}. Value remains unchanged.")       
                    

        # for mv in market_values:
        #     market_date = datetime.fromtimestamp(mv['x'] / 1000, tz=timezone.utc).date()
        #     market_date =market_date -relativedelta(years=15)
        #     market_year = market_date.year
        
        #     mv['age'] = int(mv['age'])
            
        #     try:
        #         # Adjust the market value using the cpi library

        #         mv['y'] = cpi.inflate(mv['y'], market_year)
                
        #     except Exception as e:
        #         # If any error occurs, leave the value unchanged and print the error
        #         print(f"Error adjusting for inflation for the year {market_year}: {e}. Value remains unchanged.")            
            
            
        predictions = []
        predictions_sofifa = []

        for i, mv in enumerate(market_values):
            filtered_data = market_values[:i + 1]

            year_of_data_point = datetime.utcfromtimestamp(mv['x'] / 1000).year
            age_at_this_point = mv['age']
            if mv['y'] == 0 and i > 0:
                latest_value = market_values[i - 1]['y']
            else:
                latest_value = mv['y']
            number_of_changes = len(filtered_data)
            mean_value = sum(entry['y'] for entry in filtered_data) / len(filtered_data)
            median_value = sorted(entry['y'] for entry in filtered_data)[len(filtered_data) // 2]
            std_deviation = pd.Series([entry['y'] for entry in filtered_data]).std()
            if pd.isna(std_deviation):
                std_deviation = 0

            initial_value = 0
            total_increase = sum(max(0, filtered_data[j]['y'] - (filtered_data[j - 1]['y'] if j > 0 else initial_value)) for j in range(len(filtered_data)))
            total_decrease = sum(max(0, (filtered_data[j - 1]['y'] if j > 0 else initial_value) - filtered_data[j]['y']) for j in range(len(filtered_data)))

            highest_value = max(entry['y'] for entry in market_values)
            age_at_highest_value = next(entry['age'] for entry in market_values if entry['y'] == highest_value)
            current_to_max_ratio = latest_value / highest_value if highest_value != 0 else 0

            # max_value_dates = [entry['x'] for entry in filtered_data if entry['y'] == highest_value]
            # start_date_of_max_value = min(max_value_dates) if max_value_dates else None
            # end_date_of_max_value = None

            # if latest_value == highest_value:
            #     end_date_of_max_value = datetime.now().timestamp() * 1000
            # else:
            #     indices_of_max_value = [i for i, entry in enumerate(filtered_data) if entry['y'] == highest_value]
            #     max_value_last_index = max(indices_of_max_value) if indices_of_max_value else -1

            #     if max_value_last_index < len(filtered_data) - 1:
            #         end_date_of_max_value = filtered_data[max_value_last_index + 1]['x']
            #     else:
            #         end_date_of_max_value = datetime.now().timestamp() * 1000

            # if start_date_of_max_value and end_date_of_max_value:
            #     duration_at_max_value = (end_date_of_max_value - start_date_of_max_value) / (1000 * 60 * 60 * 24)
            # else:
            #     duration_at_max_value = 0

            recent_values = [entry['y'] for entry in filtered_data[-5:]]
            changes = [recent_values[k] - recent_values[k - 1] for k in range(1, len(recent_values))]

            if highest_value > 50000000:
                significant_change_threshold = 0.02
            elif highest_value > 5000000:
                significant_change_threshold = 0.05
            else:
                significant_change_threshold = 0.10

            average_change = sum(changes) / len(changes) if changes else 0
            if abs(average_change) / highest_value <= significant_change_threshold:
                trend = "stable"
            elif average_change > 0:
                trend = "increasing"
            else:
                trend = "decreasing"

            input_data = pd.DataFrame({
                'Height': [int(scraped_data['height'])],
                'Country': [data['Country']],
                'CAPS': [int(scraped_data['caps'])],
                'CAPS GOALS': [int(scraped_data['goals'])],
                'Foot': [scraped_data['foot']],
                'Age': [int(age_at_this_point)],
                'Position': [scraped_data['position']],
                'Position Role': [scraped_data['position_role']],
                'MarketValue': [float(latest_value)],
                'DaysLeftofContract': [float(scraped_data['days_left_contract'])],
                'TEAM RATING': [float(data['TEAM RATING'])],
                'TEAM RANKING': [int(data['TEAM RANKING'])],
                'LEAGUE RATING': [float(data['LEAGUE RATING'])],
                'HighestMarketValue': [highest_value],
                'AgeAtHighestMarketValue': [int(age_at_highest_value)],
                'NumberOfMarketValueChanges': [int(number_of_changes)],
                'LatestMarketValue': [float(latest_value)],
                'MeanMarketValue': [float(mean_value)],
                'MedianMarketValue': [float(median_value)],
                'MarketValueStdDeviation': [float(std_deviation)],
                'TotalIncrease': [float(total_increase)],
                'TotalDecrease': [float(total_decrease)],
                'CurrentToMaxRatio': [float(current_to_max_ratio)],
                'DurationAtMaxValue': [float(scraped_data['duration_at_max_value'])],
                'MarketValueTrend': [trend]
            })

            try:
                prediction = make_prediction(input_data)
                prediction_sofifa = make_prediction_sofifa(input_data)
                predictions.append((prediction, input_data, year_of_data_point))
                predictions_sofifa.append((prediction_sofifa, input_data, year_of_data_point))
            except Exception as e:
                continue

        # Prepare data for exporting predictions
        if predictions:
            best_prediction, best_data, best_prime_year = max(predictions, key=lambda x: x[0])



        if predictions_sofifa:
            best_prediction_sofifa, best_data_sofifa, best_prime_year_sofifa = max(predictions_sofifa, key=lambda x: x[0])



        years = [pred[2] for pred in predictions]
        prediction_values = [pred[0] for pred in predictions]
        years_sofifa = [pred[2] for pred in predictions_sofifa]
        prediction_values_sofifa = [pred[0] for pred in predictions_sofifa]

        highest_prediction = max(predictions, key=lambda x: x[0])
        highest_prediction_sofifa = max(predictions_sofifa, key=lambda x: x[0])

        return render_template('prime-result.html', 
                               prediction=best_prediction, 
                               prediction_sofifa=best_prediction_sofifa, 
                               form_data=data, 
                               scraped_data=scraped_data,
                               number_of_changes=best_data['NumberOfMarketValueChanges'][0], 
                               mean_value=best_data['MeanMarketValue'][0], 
                               median_value=best_data['MedianMarketValue'][0], 
                               std_deviation=best_data['MarketValueStdDeviation'][0], 
                               total_increase=best_data['TotalIncrease'][0], 
                               current_to_max_ratio=best_data['CurrentToMaxRatio'][0], 
                               trend=best_data['MarketValueTrend'][0], 
                               total_decrease=best_data['TotalDecrease'][0], 
                               highest_value=best_data['HighestMarketValue'][0], 
                               age_at_highest_value=best_data['AgeAtHighestMarketValue'][0], 
                               prime_year=best_prime_year, 
                               prime_year_sofifa=best_prime_year_sofifa,
                               team_name=data['team_name'],
                               years=years, 
                               prediction_values=prediction_values,
                               years_sofifa=years_sofifa,
                               prediction_values_sofifa=prediction_values_sofifa,
                               highest_prediction=highest_prediction[0],
                               highest_prediction_sofifa=highest_prediction_sofifa[0],
                               latest_value=best_data['LatestMarketValue'][0],
                               age=best_data['Age'][0],
                               duration_at_max_value=duration_at_max_value,
                               latest_value_sofifa=best_data_sofifa['LatestMarketValue'][0],
                               age_sofifa=best_data_sofifa['Age'][0],
                               league_img=league_img)
    except Exception as e:
        print(f"Error: {e}")
        return 'FAILED'

    
@app.route('/get_leagues', methods=['GET'])
def get_leagues():
    leagues = ranking_mapping.groupby(['country_league'])['league_name'].apply(list).to_dict()
    return jsonify({'leagues': leagues})

@app.route('/get_clubs/<league>', methods=['GET'])
def get_clubs(league):
    try:
        league_id = ranking_mapping.loc[ranking_mapping['league_name'] == league, 'tmcl'].values[0]
        league_clubs = [club for club in clubs_data if club['tmcl'] == league_id]
        if not league_clubs:
            print(f"No clubs found for league: {league} with league_id: {league_id}")
            return jsonify({'clubs': []})
        league_clubs_sorted = sorted(league_clubs, key=lambda x: x['contestantName'])
        return jsonify({'clubs': league_clubs_sorted})
    except Exception as e:
        print(f"Error fetching clubs for league {league}: {e}")
        return jsonify({'clubs': []})


# Implement the scrape_transfermarkt function to extract player data
def get_team_shield_url(team_id):
    return f"https://cdn.sportfeeds.io/sdl/images/team/crest/large/{team_id}.png"

def scrape_transfermarkt(url):
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        player_name = None
        headline_wrapper = soup.find('h1', class_='data-header__headline-wrapper')
        if headline_wrapper:
            # Extract text nodes and text within strong tags
            text_nodes = headline_wrapper.find_all(text=True, recursive=False)
            strong_tag = headline_wrapper.find('strong')
            if strong_tag:
                text_nodes.append(strong_tag.get_text(strip=True))

            player_name_parts = []
            for part in text_nodes:
                text = part.strip().replace('#', '').strip()
                if text and not text.isdigit():
                    player_name_parts.append(text)

            # Join the parts and remove extra spaces
            player_name = ' '.join(player_name_parts).strip()

        # Print player name for debugging
        print(f"Extracted player name: {player_name}")
        
        # Extracting height
        height = None
        ul_tags = soup.find_all('ul', class_='data-header__items')
        if len(ul_tags) >= 2:
            second_ul = ul_tags[1]
            for li in second_ul.find_all('li', class_='data-header__label'):
                if 'Height:' in li.text:
                    height_text = li.find_next('span').text.strip()
                    if 'N/A' in height_text:
                        height = 180  # Set default height if the value is N/A
                    else:
                        height = int(height_text.replace(' m', '').replace(',', ''))
        
        if height is None or height == 'None':
            height =180

        # Extracting age
        age = None
        if len(ul_tags) >= 1:
            first_ul = ul_tags[0]
            for li in first_ul.find_all('li', class_='data-header__label'):
                if 'Date of birth/Age:' in li.text:
                    age_text = li.find_next('span').text.strip()
                    try:
                        age = int(age_text.split('(')[1].replace(')', '').strip())
                    except (IndexError, ValueError):
                        print(f"Failed to extract age from text: {age_text}")

        if age is None or age == 'None':
            age =25

        age = int(age)
        # Extracting position
        position = 'Central Midfield'
        position_role = 'Midfield'

        if len(ul_tags) >= 2:
            second_ul = ul_tags[1]
            for li in second_ul.find_all('li', class_='data-header__label'):
                if 'Position:' in li.text:
                    position = li.find_next('span').text.strip()
                    if position == 'Defender':
                        position = 'Centre-Back'
                    if position == 'Striker':
                        position = 'Centre-Forward'
                    if position == 'Midfielder':
                        position = 'Central Midfield'                

                    if position in ['Goalkeeper']:
                        position_role = 'Goalkeeper'
                    elif position in ['Right-Back', 'Left-Back']:
                        position_role = 'Full-Back'
                    elif position == 'Centre-Back':
                        position_role = 'Centre-Back'
                    elif position in ['Right Midfield', 'Left Midfield', 'Central Midfield', 'Defensive Midfield', 'Attacking Midfield']:
                        position_role = 'Midfield'
                    elif position in ['Right Winger', 'Left Winger']:
                        position_role = 'Winger'
                    elif position in ['Centre-Forward', 'Second Striker']:
                        position_role = 'Forward'
                        break
       
        # Extracting foot
        foot = 'right'  # Default value
        info_table_div = soup.find('div', class_='info-table info-table--right-space')
        if info_table_div:
            for span in info_table_div.find_all('span', class_='info-table__content info-table__content--regular'):
                if 'Foot:' in span.text:
                    foot = span.find_next('span', class_='info-table__content--bold').text.strip()
        if foot =='both':
            foot = 'right'

        # Extracting CAPS and GOALS
        caps, goals = 0, 0
        for li in soup.find_all('li', class_='data-header__label'):
            # Check for 'Current international:' and youth divisions
            if 'Current international:' in li.text:
                current_int = li.find('a')
                if current_int:
                    country_info = current_int.text.strip()
                    if any(f'U{n}' in country_info for n in range(15, 24)):  # Check for U15 to U23
                        # If it's a youth division, set CAPS and GOALS to 0 and break the loop
                        caps, goals = 0, 0
                        break

            # Check for 'Caps/Goals:' and extract values
            if 'Caps/Goals:' in li.text:
                caps_goals = li.find_all('a', class_='data-header__content--highlight')
                if caps_goals and len(caps_goals) >= 2:
                    caps = int(caps_goals[0].text.strip())
                    goals = int(caps_goals[1].text.strip())

        # Extracting contract expiration date and days left
        contract_date = None
        contract_expire = soup.find("span", string="Contract expires:")
        if contract_expire:
            contract_date_span = contract_expire.find_next("span", class_="info-table__content--bold")
            if contract_date_span:
                contract_date_text = contract_date_span.text.strip()
                if contract_date_text != '-':
                    try:
                        contract_date = pd.to_datetime(contract_date_text, format='%b %d, %Y')
                    except Exception as e:
                        print(f"Error parsing contract date: {e}")

        if not contract_date:
            today = datetime.today()
            next_year_jan_1 = datetime(today.year + 1, 1, 1)
            next_year_jul_1 = datetime(today.year + 1, 7, 1)
            days_left_contract = max((next_year_jan_1 - today).days, (next_year_jul_1 - today).days)
        else:
            days_left_contract = (contract_date - datetime.today()).days
       
       
        # Extracting profile image
        profile_img = None
        profile_img_div = soup.find('div', class_='data-header__profile-container')
        if profile_img_div:
            img_tag = profile_img_div.find('img')
            if img_tag and 'src' in img_tag.attrs:
                profile_img = img_tag['src'].replace('header', 'big')

        profile_img = remove_background(profile_img)
        if profile_img:
            profile_img = f"data:image/png;base64,{profile_img}"
        else:
            profile_img = None
        # Extracting market value and historical data

        country = 'Spain'
        nationality_el = soup.find("span", {"itemprop": "nationality"})
        if nationality_el:
            country = nationality_el.getText().replace("\n", "").strip()
       
        player_id = url.split('/')[-1]

        api_url = f"https://www.transfermarkt.com/ceapi/marketValueDevelopment/graph/{player_id}"
        response = requests.get(api_url, headers=HEADERS)
        data = response.json()

        if response.status_code != 200:
            return 'FAILED'

        fallback_value = 10000  # Default fallback value for market value
        if not data.get('list') or (data.get('current') == '-' and data.get('highest') == '-'):
            market_value = fallback_value
            return {
                'height': height,
                'age': age,
                'position': position,
                'position_role': position_role,
                'foot': foot,
                'caps': caps,
                'goals': goals,
                'MarketValue': market_value,
                'days_left_contract': days_left_contract,
                'highest_market_value': market_value,
                'age_at_highest_value': age,
                'number_of_changes': 1,
                'latest_value': market_value,
                'mean_value': market_value,
                'median_value': market_value,
                'std_deviation': 0,
                'total_increase': 0,
                'total_decrease': 0,
                'current_to_max_ratio': 1,
                'duration_at_max_value': 0,
                'trend': 'stable',
                'profile_img': profile_img,
                'player_name': player_name,
                'nationality': country
            }

        # Use today's date for the cutoff date
        today = datetime.today()
        cutoff_date = today.timestamp() * 1000
        for entry in data['list']:
            if entry['verein'] == "Retired":
                cutoff_date = entry['x']
                age = entry['age']
                break

        filtered_data = [entry for entry in data['list'] if entry['x'] <= cutoff_date]

        if not filtered_data:
            market_value = fallback_value
            return {
                'height': height,
                'age': age,
                'position': position,
                'position_role': position_role,
                'foot': foot,
                'caps': caps,
                'goals': goals,
                'MarketValue': market_value,
                'days_left_contract': days_left_contract,
                'highest_market_value': market_value,
                'age_at_highest_value': age,
                'number_of_changes': 1,
                'latest_value': market_value,
                'mean_value': market_value,
                'median_value': market_value,
                'std_deviation': 0,
                'total_increase': 0,
                'total_decrease': 0,
                'current_to_max_ratio': 1,
                'duration_at_max_value': 0,
                'trend': 'stable',
                'profile_img': profile_img,
                'player_name': player_name,
                'nationality': country
            }

        # Replace '-' with the last valid market value
        for i in range(1, len(filtered_data)):
            if filtered_data[i]['y'] == 0:
                filtered_data[i]['y'] = filtered_data[i-1]['y']

        # Extracting key metrics from filtered data
        highest_value = max(entry['y'] for entry in filtered_data)
        age_at_highest_value = next(entry['age'] for entry in filtered_data if entry['y'] == highest_value)
        number_of_changes = len(filtered_data)
        latest_value = filtered_data[-1]['y']

        # Calculating summary statistics
        values = [entry['y'] for entry in filtered_data]
        mean_value = sum(values) / len(values)
        median_value = sorted(values)[len(values) // 2]
        std_deviation = pd.Series(values).std()

        initial_value = 0  # Assume the player starts from zero market value
        total_increase = sum(max(0, values[i] - (values[i - 1] if i > 0 else initial_value)) for i in range(len(values)))
        total_decrease = sum(max(0, (values[i - 1] if i > 0 else initial_value) - values[i]) for i in range(len(values)))

        # Calculating ratio of current to maximum market value
        current_to_max_ratio = latest_value / highest_value if highest_value != 0 else 0

        # Calculating duration at maximum market value
        max_value_dates = [entry['x'] for entry in filtered_data if entry['y'] == highest_value]
        start_date_of_max_value = min(max_value_dates) if max_value_dates else None
        end_date_of_max_value = None

        if latest_value == highest_value:
            end_date_of_max_value = cutoff_date  # Use cutoff date if the current value is still at its peak
        else:
            indices_of_max_value = [i for i, entry in enumerate(filtered_data) if entry['y'] == highest_value]
            max_value_last_index = max(indices_of_max_value) if indices_of_max_value else -1

            if max_value_last_index < len(filtered_data) - 1:
                end_date_of_max_value = filtered_data[max_value_last_index + 1]['x']
            else:
                end_date_of_max_value = cutoff_date  # Use cutoff date if there's no next value after the last max value

        if start_date_of_max_value and end_date_of_max_value:
            duration_at_max_value = (end_date_of_max_value - start_date_of_max_value) / (1000 * 60 * 60 * 24)
        else:
            duration_at_max_value = 0  # Set duration to 0 if no max value dates are found

        # Determining the trend based on recent changes
        recent_values = values[-5:]  # Consider the last 5 updates
        changes = [recent_values[i] - recent_values[i - 1] for i in range(1, len(recent_values))]

        # Define threshold based on highest value
        if highest_value > 50000000:  # High-value players
            significant_change_threshold = 0.02
        elif highest_value > 5000000:  # Mid-value players
            significant_change_threshold = 0.05
        else:  # Low-value players
            significant_change_threshold = 0.10

        average_change = sum(changes) / len(changes) if changes else 0
        if abs(average_change) / latest_value <= significant_change_threshold:
            trend = "stable"
        elif average_change > 0:
            trend = "increasing"
        else:
            trend = "decreasing"

        return {
            'height': height,
            'age': age,
            'position': position,
            'position_role': position_role,
            'foot': foot,
            'caps': caps,
            'goals': goals,
            'MarketValue': latest_value,
            'days_left_contract': days_left_contract,
            'highest_market_value': highest_value,
            'age_at_highest_value': age_at_highest_value,
            'number_of_changes': number_of_changes,
            'latest_value': latest_value,
            'mean_value': mean_value,
            'median_value': median_value,
            'std_deviation': std_deviation,
            'total_increase': total_increase,
            'total_decrease': total_decrease,
            'current_to_max_ratio': current_to_max_ratio,
            'duration_at_max_value': duration_at_max_value,
            'trend': trend,
            'profile_img': profile_img,
            'player_name': player_name,
            'nationality': country,
        }

    except Exception as e:
        print(f"Error scraping Transfermarkt: {e}")
        return None




if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


